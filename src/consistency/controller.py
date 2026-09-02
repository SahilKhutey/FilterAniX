"""Temporal Decision Controller for Keyframe and Conditioning Management."""
from typing import Optional
import numpy as np

from src.consistency.types import TemporalState, RenderDecision, ReferenceProfile
from src.consistency.scene import SceneDetector
from src.consistency.motion import MotionAnalyzer
from src.consistency.identity import IdentityScorer
from src.vision.models import MotionData


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
        """Determines if a frame is a scene cut, keyframe, or intermediate temporal neighbor."""
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
            # First frame or Scene Cut Reset
            self.state.scene_id = scene_id
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            self.state.stable_count = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                is_scene_cut=is_scene_cut,
                is_keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=False,
                similarity_warning=similarity_warning,
                reason="scene_cut_reset" if is_scene_cut else "initial_keyframe",
            )

        self.state.frames_since_keyframe += 1

        if is_high_motion:
            # Rapid gesture / sudden movement
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                is_scene_cut=False,
                is_keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=False,
                similarity_warning=similarity_warning,
                reason="high_motion_keyframe",
            )

        if self.state.frames_since_keyframe >= self.keyframe_interval:
            # Regular anchor interval
            self.state.last_keyframe_idx = frame_index
            self.state.frames_since_keyframe = 0
            return RenderDecision(
                frame_index=frame_index,
                timestamp=timestamp,
                scene_id=scene_id,
                is_scene_cut=False,
                is_keyframe=True,
                motion_score=motion_score,
                reference_strength=self.ref_strength_kf,
                preserve_previous=True,
                similarity_warning=similarity_warning,
                reason="interval_keyframe",
            )

        # Standard intermediate neighbor frame
        return RenderDecision(
            frame_index=frame_index,
            timestamp=timestamp,
            scene_id=scene_id,
            is_scene_cut=False,
            is_keyframe=False,
            motion_score=motion_score,
            reference_strength=self.ref_strength_inter,
            preserve_previous=True,
            similarity_warning=similarity_warning,
            reason="temporal_neighbor",
        )

    def reset(self):
        self.scene_detector.reset()
        self.state = TemporalState()
