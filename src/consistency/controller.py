from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from src.consistency.types import TemporalState, ReferenceProfile
from src.consistency.scene import SceneDetector
from src.consistency.motion import MotionAnalyzer
from src.consistency.identity import IdentityScorer
from src.vision.models import MotionData


@dataclass
class RenderDecision:
    frame_index: int
    scene_id: int

    keyframe: bool
    scene_cut: bool

    motion_score: float

    reference_strength: float

    reason: str

    # Backwards-compatibility properties / attributes
    timestamp: float = 0.0
    preserve_previous: bool = True
    similarity_warning: bool = False

    @property
    def is_keyframe(self) -> bool:
        return self.keyframe

    @property
    def is_scene_cut(self) -> bool:
        return self.scene_cut

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "scene_id": self.scene_id,
            "keyframe": self.keyframe,
            "scene_cut": self.scene_cut,
            "motion_score": round(self.motion_score, 4),
            "reference_strength": round(self.reference_strength, 3),
            "preserve_previous": self.preserve_previous,
            "similarity_warning": self.similarity_warning,
            "reason": self.reason,
        }


class IdentityRenderController:
    """Decides keyframe scheduling, scene transitions, and reference conditioning strengths."""

    def __init__(
        self,
        keyframe_interval: int = 12,
        motion_threshold: float = 0.22,
    ):
        self.keyframe_interval = keyframe_interval
        self.motion_threshold = motion_threshold

    def decide(
        self,
        frame_index: int,
        scene_id: int,
        scene_cut: bool,
        motion_score: float,
    ) -> RenderDecision:
        if scene_cut:
            return RenderDecision(
                frame_index=frame_index,
                scene_id=scene_id,
                keyframe=True,
                scene_cut=True,
                motion_score=motion_score,
                reference_strength=0.85,
                preserve_previous=False,
                reason="scene_cut",
            )

        if frame_index == 0:
            return RenderDecision(
                frame_index=frame_index,
                scene_id=scene_id,
                keyframe=True,
                scene_cut=False,
                motion_score=motion_score,
                reference_strength=0.85,
                preserve_previous=False,
                reason="first_frame",
            )

        if (
            frame_index
            % self.keyframe_interval
            == 0
        ):
            return RenderDecision(
                frame_index=frame_index,
                scene_id=scene_id,
                keyframe=True,
                scene_cut=False,
                motion_score=motion_score,
                reference_strength=0.75,
                preserve_previous=True,
                reason="scheduled_keyframe",
            )

        if (
            motion_score
            >= self.motion_threshold
        ):
            return RenderDecision(
                frame_index=frame_index,
                scene_id=scene_id,
                keyframe=True,
                scene_cut=False,
                motion_score=motion_score,
                reference_strength=0.70,
                preserve_previous=False,
                reason="high_motion",
            )

        return RenderDecision(
            frame_index=frame_index,
            scene_id=scene_id,
            keyframe=False,
            scene_cut=False,
            motion_score=motion_score,
            reference_strength=0.55,
            preserve_previous=True,
            reason="intermediate",
        )


class TemporalController:
    """Evaluates frame conditions and generates explicit per-frame RenderDecision instructions."""

    def __init__(
        self,
        keyframe_interval: int = 12,
        keyframe_motion_threshold: float = 0.28,
        reference_strength_keyframe: float = 0.85,
        reference_strength_intermediate: float = 0.55,
        reference_profile: Optional[ReferenceProfile] = None,
    ):
        self.keyframe_interval = keyframe_interval
        self.keyframe_motion_threshold = keyframe_motion_threshold
        self.ref_strength_kf = reference_strength_keyframe
        self.ref_strength_inter = reference_strength_intermediate

        self.scene_detector = SceneDetector()
        self.motion_analyzer = MotionAnalyzer(keyframe_motion_threshold=keyframe_motion_threshold)
        self.identity_scorer = IdentityScorer(reference_profile) if reference_profile else None
        self.state = TemporalState()

    def evaluate_frame(
        self,
        frame_index: int,
        timestamp: float,
        frame_rgb: np.ndarray,
        motion_data: Optional[MotionData] = None,
    ) -> RenderDecision:
        # 1. Check Scene Cut
        is_scene_cut, scene_id = self.scene_detector.process(frame_rgb)

        # 2. Check Motion Energy
        motion_score = self.motion_analyzer.calculate_score(motion_data)
        is_high_motion = self.motion_analyzer.is_high_motion(motion_score)

        # 3. Check Similarity Warning
        similarity_warning = False
        if self.identity_scorer is not None:
            metrics = self.identity_scorer.evaluate_frame(frame_rgb)
            similarity_warning = metrics.warning
            self.state.last_similarity = metrics.similarity

        # 4. Decision Logic
        if frame_index == 0 or is_scene_cut:
            self.state.scene_id = scene_id
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            self.state.stable_count = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                scene_cut=is_scene_cut,
                keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=False,
                similarity_warning=similarity_warning,
                reason="scene_cut_reset" if is_scene_cut else "initial_keyframe",
            )

        self.state.frames_since_keyframe += 1

        if is_high_motion:
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                scene_cut=False,
                keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=False,
                similarity_warning=similarity_warning,
                reason="high_motion_keyframe",
            )

        if self.state.frames_since_keyframe >= self.keyframe_interval:
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                scene_cut=False,
                keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=True,
                similarity_warning=similarity_warning,
                reason="interval_keyframe",
            )

        return RenderDecision(
            frame_index=frame_index,
            timestamp=timestamp,
            scene_id=scene_id,
            scene_cut=False,
            keyframe=False,
            motion_score=motion_score,
            reference_strength=self.ref_strength_inter,
            preserve_previous=True,
            similarity_warning=similarity_warning,
            reason="temporal_neighbor",
        )

    def reset(self):
        self.scene_detector.reset()
        self.state = TemporalState()
