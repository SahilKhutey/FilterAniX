"""Motion Analysis and High-Motion Keyframe Candidate Scoring."""
from typing import Optional
import numpy as np

from src.vision.models import MotionData


class MotionAnalyzer:
    """Computes normalized motion score and identifies keyframe candidate intervals."""

    def __init__(self, keyframe_motion_threshold: float = 0.28):
        self.keyframe_motion_threshold = keyframe_motion_threshold

    def calculate_score(self, motion_data: Optional[MotionData] = None) -> float:
        """Computes a normalized motion score in [0.0, 1.0]."""
        if not motion_data or not motion_data.valid:
            return 0.0

        # Combine magnitude and moving pixel coverage
        mag_norm = min(1.0, motion_data.mean_magnitude / 10.0)
        moving_norm = min(1.0, motion_data.moving_pixel_ratio * 3.0)

        score = 0.6 * mag_norm + 0.4 * moving_norm
        return float(min(1.0, max(0.0, score)))

    def is_high_motion(self, score: float) -> bool:
        return score >= self.keyframe_motion_threshold
