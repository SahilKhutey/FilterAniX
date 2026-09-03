from __future__ import annotations

import cv2
import numpy as np


class FramePropagator:

    def __init__(
        self,
        blend: float = 0.20,
    ):
        self.blend = float(
            np.clip(blend, 0.0, 1.0)
        )

    @staticmethod
    def _flow(
        previous_rgb: np.ndarray,
        current_rgb: np.ndarray,
    ) -> np.ndarray:

        previous_gray = cv2.cvtColor(
            previous_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        current_gray = cv2.cvtColor(
            current_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        return cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

    def warp(
        self,
        previous_source: np.ndarray,
        current_source: np.ndarray,
        previous_art: np.ndarray,
    ) -> np.ndarray:

        flow = self._flow(
            previous_source,
            current_source,
        )

        height, width = current_source.shape[:2]

        grid_x, grid_y = np.meshgrid(
            np.arange(width),
            np.arange(height),
        )

        map_x = (
            grid_x.astype(np.float32)
            - flow[..., 0]
        )

        map_y = (
            grid_y.astype(np.float32)
            - flow[..., 1]
        )

        warped = cv2.remap(
            previous_art,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return warped

    def blend_frames(
        self,
        propagated: np.ndarray,
        fresh: np.ndarray,
    ) -> np.ndarray:

        return cv2.addWeighted(
            fresh,
            1.0 - self.blend,
            propagated,
            self.blend,
            0.0,
        )
