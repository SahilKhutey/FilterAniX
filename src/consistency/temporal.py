from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


class CharacterTemporalState:
    """Maintains consistent character appearance and color across intermediate frames within a scene."""

    def __init__(
        self,
        blend_strength: float = 0.16,
    ):
        self.blend_strength = blend_strength
        self.previous: Optional[np.ndarray] = None
        self.keyframe: Optional[np.ndarray] = None
        self.scene_id: Optional[int] = None

    def reset(self) -> None:
        """Hard reset across scene cuts or sequence restarts."""
        self.previous = None
        self.keyframe = None
        self.scene_id = None

    def set_keyframe(
        self,
        frame: np.ndarray,
        scene_id: int,
    ) -> None:
        """Sets anchor keyframe for the current scene."""
        self.keyframe = frame.copy()
        self.previous = frame.copy()
        self.scene_id = scene_id

    def propagate(
        self,
        frame: np.ndarray,
        scene_id: int,
    ) -> np.ndarray:
        """Propagates character style to an intermediate frame without cross-scene bleeding."""
        if (
            self.previous is None
            or self.scene_id != scene_id
        ):
            self.previous = frame.copy()
            self.scene_id = scene_id
            return frame

        result = cv2.addWeighted(
            frame,
            1.0 - self.blend_strength,
            self.previous,
            self.blend_strength,
            0,
        )

        self.previous = result.copy()
        return result
