"""Temporal Deflicker and Rolling Window Luminance Regularization."""
from collections import deque
from typing import Deque, Optional
import cv2
import numpy as np
from filteranix.core.config import TemporalConfig


class TemporalDeflicker:
    """Suppresses high-frequency temporal luminance and chrominance oscillations."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.window_size = max(1, self.config.temporal_window_size)
        self.history: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def process(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Applies rolling temporal exponential/moving-average smoothing to eliminate intensity flutter."""
        if not self.config.enable_deflicker or self.window_size <= 1:
            return frame_rgb

        self.history.append(frame_rgb.astype(np.float32))

        if len(self.history) == 1:
            return frame_rgb

        # Compute Gaussian-weighted temporal average over window
        weights = np.exp(-0.5 * (np.arange(len(self.history)) - (len(self.history) - 1)) ** 2)
        weights /= np.sum(weights)

        stacked = np.stack(list(self.history), axis=0)  # (N, H, W, 3)
        averaged = np.sum(stacked * weights[:, np.newaxis, np.newaxis, np.newaxis], axis=0)

        return np.clip(averaged, 0, 255).astype(np.uint8)
