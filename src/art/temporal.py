"""Temporal Stabilizer and Scene Cut Detector."""
from collections import deque
from typing import Deque, Optional
import cv2
import numpy as np


class TemporalStabilizer:
    """Maintains frame-to-frame visual continuity while detecting and handling scene cuts."""

    def __init__(self, alpha: float = 0.60, scene_cut_threshold: float = 0.40, window_size: int = 4):
        self.alpha = alpha
        self.scene_cut_threshold = scene_cut_threshold
        self.window_size = window_size
        self.prev_raw_frame: Optional[np.ndarray] = None
        self.prev_art_frame: Optional[np.ndarray] = None
        self.history: Deque[np.ndarray] = deque(maxlen=window_size)

    def is_scene_cut(self, curr_raw: np.ndarray) -> bool:
        """Determines if a camera shot transition / hard cut occurred between frames."""
        if self.prev_raw_frame is None:
            return False

        # Compute normalized Mean Squared Error on downscaled luminance
        g_curr = cv2.resize(cv2.cvtColor(curr_raw, cv2.COLOR_RGB2GRAY), (128, 72))
        g_prev = cv2.resize(cv2.cvtColor(self.prev_raw_frame, cv2.COLOR_RGB2GRAY), (128, 72))

        diff = np.abs(g_curr.astype(np.float32) - g_prev.astype(np.float32)) / 255.0
        mse = float(np.mean(diff ** 2))

        # Normalized color histogram correlation
        hist_curr = cv2.calcHist([curr_raw], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist_prev = cv2.calcHist([self.prev_raw_frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist_curr, hist_curr)
        cv2.normalize(hist_prev, hist_prev)
        hist_corr = float(cv2.compareHist(hist_curr, hist_prev, cv2.HISTCMP_CORREL))

        # Scene cut detected if MSE is high and histogram correlation drops
        return mse > self.scene_cut_threshold or hist_corr < 0.45

    def stabilize(self, curr_raw: np.ndarray, curr_art: np.ndarray) -> np.ndarray:
        """Stabilizes stylized output frame against temporal oscillations and flicker."""
        if self.prev_raw_frame is None or self.prev_art_frame is None or self.is_scene_cut(curr_raw):
            # Reset history on first frame or scene cut
            self.history.clear()
            self.prev_raw_frame = curr_raw.copy()
            self.prev_art_frame = curr_art.copy()
            self.history.append(curr_art.astype(np.float32))
            return curr_art

        # Temporal Alpha Blend with previous stylized frame
        curr_f = curr_art.astype(np.float32)
        prev_f = self.prev_art_frame.astype(np.float32)

        blended = (1.0 - self.alpha) * curr_f + self.alpha * prev_f
        self.history.append(blended)

        # Multi-frame rolling window average
        stacked = np.stack(list(self.history), axis=0)
        smoothed = np.mean(stacked, axis=0)

        final_art = np.clip(smoothed, 0, 255).astype(np.uint8)

        self.prev_raw_frame = curr_raw.copy()
        self.prev_art_frame = final_art.copy()

        return final_art

    def reset(self):
        self.prev_raw_frame = None
        self.prev_art_frame = None
        self.history.clear()
