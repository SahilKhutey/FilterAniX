"""Scene Transition & Cut Detector."""
from typing import Optional, Tuple
import cv2
import numpy as np


class SceneDetector:
    """Monitors consecutive frames for scene transitions and increments scene IDs."""

    def __init__(self, mse_threshold: float = 0.40, hist_threshold: float = 0.45):
        self.mse_threshold = mse_threshold
        self.hist_threshold = hist_threshold
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_hist: Optional[np.ndarray] = None
        self.current_scene_id: int = 0

    def process(self, frame_rgb: np.ndarray) -> Tuple[bool, int]:
        """Evaluates whether current frame is a scene cut.
        
        Returns:
            is_scene_cut: bool
            scene_id: int
        """
        curr_gray = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), (128, 72))
        curr_hist = cv2.calcHist([frame_rgb], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(curr_hist, curr_hist)

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_hist = curr_hist
            return False, self.current_scene_id

        # 1. Normalized Mean Squared Error
        diff = np.abs(curr_gray.astype(np.float32) - self.prev_gray.astype(np.float32)) / 255.0
        mse = float(np.mean(diff ** 2))

        # 2. Histogram Correlation
        hist_corr = float(cv2.compareHist(curr_hist, self.prev_hist, cv2.HISTCMP_CORREL))

        is_cut = (mse > self.mse_threshold) or (hist_corr < self.hist_threshold)

        if is_cut:
            self.current_scene_id += 1

        self.prev_gray = curr_gray
        self.prev_hist = curr_hist

        return is_cut, self.current_scene_id

    def reset(self):
        self.prev_gray = None
        self.prev_hist = None
        self.current_scene_id = 0
