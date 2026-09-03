from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class KeyframeDecision:
    frame_index: int
    scene_id: int
    is_keyframe: bool
    is_scene_cut: bool
    motion_score: float
    reference_strength: float
    reason: str


class KeyframeScheduler:
    """
    Converts the Phase-4 temporal plan into rendering decisions.

    Phase 4 remains authoritative.

    We only add mandatory guarantees:
      - first frame
      - scene cuts
      - scene changes
      - sufficiently separated high-motion frames
    """

    def __init__(
        self,
        minimum_interval: int = 12,
        motion_threshold: float = 0.22,
    ):
        self.minimum_interval = max(1, int(minimum_interval))
        self.motion_threshold = float(motion_threshold)

    def build(
        self,
        temporal_plan: Iterable[dict[str, Any]],
    ) -> list[KeyframeDecision]:

        decisions: list[KeyframeDecision] = []

        last_keyframe = -10**9
        previous_scene: int | None = None

        for position, item in enumerate(temporal_plan):

            frame_index = int(
                item.get("frame_index", position)
            )

            scene_id = int(
                item.get("scene_id", 0)
            )

            scene_cut = bool(
                item.get(
                    "is_scene_cut",
                    item.get("scene_cut", False),
                )
            )

            requested_keyframe = bool(
                item.get(
                    "is_keyframe",
                    item.get("keyframe", False),
                )
            )

            motion_score = float(
                item.get("motion_score", 0.0)
            )

            reference_strength = float(
                item.get("reference_strength", 0.55)
            )

            scene_change = (
                previous_scene is not None
                and scene_id != previous_scene
            )

            mandatory = (
                frame_index == 0
                or scene_cut
                or scene_change
            )

            high_motion = (
                motion_score >= self.motion_threshold
            )

            if mandatory:

                selected = True

                if frame_index == 0:
                    reason = "initial_keyframe"
                elif scene_cut:
                    reason = "scene_cut"
                else:
                    reason = "scene_change"

            elif requested_keyframe:

                # Phase 4 explicitly requested this.
                selected = True

                reason = str(
                    item.get(
                        "reason",
                        "phase4_keyframe",
                    )
                )

            elif (
                high_motion
                and frame_index - last_keyframe
                >= self.minimum_interval
            ):

                selected = True
                reason = "high_motion"

            else:

                selected = False
                reason = "intermediate"

            if selected:
                last_keyframe = frame_index

            decisions.append(
                KeyframeDecision(
                    frame_index=frame_index,
                    scene_id=scene_id,
                    is_keyframe=selected,
                    is_scene_cut=scene_cut,
                    motion_score=motion_score,
                    reference_strength=reference_strength,
                    reason=reason,
                )
            )

            previous_scene = scene_id

        return decisions
