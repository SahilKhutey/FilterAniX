from typing import Optional
import cv2
import numpy as np

from .types import MotionData


class OpticalFlowMotion:
    """Dense Farneback optical flow for frame-to-frame motion statistics."""

    def __init__(self):
        self.previous_gray: Optional[np.ndarray] = None

    def process(self, frame_bgr: np.ndarray) -> MotionData:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.previous_gray is None:
            self.previous_gray = gray
            return MotionData(valid=False)

        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True,
        )

        threshold = 1.5
        moving = magnitude > threshold

        result = MotionData(
            mean_magnitude=float(np.mean(magnitude)),
            mean_angle=float(np.mean(angle)),
            moving_pixel_ratio=float(np.mean(moving)),
            valid=True,
        )

        self.previous_gray = gray
        return result
