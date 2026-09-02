"""Dense Optical Flow Motion Estimation Engine."""
import math
from typing import Optional, Tuple
import cv2
import numpy as np

from src.vision.models import MotionData


class MotionEngine:
    """Computes dense optical flow, mean magnitude, dominant motion vector, and moving pixel ratio."""

    def __init__(self, movement_threshold_px: float = 1.0):
        self.movement_threshold_px = movement_threshold_px
        self.prev_gray: Optional[np.ndarray] = None

    def process(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], MotionData]:
        """Computes motion vector field between previous frame and current frame.
        
        Returns:
            flow_field: (H, W, 2) float32 (dx, dy) or None for first frame
            motion_data: MotionData summary
        """
        curr_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return None, MotionData(valid=False)

        # Compute Farneback Dense Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

        self.prev_gray = curr_gray

        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx ** 2 + dy ** 2)
        angle = (np.arctan2(dy, dx) * 180.0 / np.pi) % 360.0

        mean_mag = float(np.mean(magnitude))
        mean_ang = float(np.mean(angle))
        
        moving_pixels = np.count_nonzero(magnitude > self.movement_threshold_px)
        total_pixels = magnitude.size
        moving_ratio = float(moving_pixels) / float(total_pixels)

        mean_dx = float(np.mean(dx))
        mean_dy = float(np.mean(dy))

        motion_data = MotionData(
            mean_magnitude=mean_mag,
            mean_angle=mean_ang,
            moving_pixel_ratio=moving_ratio,
            valid=True,
            dx=mean_dx,
            dy=mean_dy,
        )

        return flow, motion_data

    def reset(self):
        self.prev_gray = None
