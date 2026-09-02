"""Temporal Viseme Smoothing and Hysteresis Filter."""
from collections import Counter
from typing import List
import numpy as np

from src.lipsync.analyzer import LipSyncRecord, VisemeState


class LipSyncSmoother:
    """Smooths raw frame-by-frame viseme timeline using rolling window filtering and ratio curve smoothing."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(3, window_size if window_size % 2 != 0 else window_size + 1)

    def smooth_timeline(self, raw_records: List[LipSyncRecord]) -> List[LipSyncRecord]:
        """Applies temporal sliding window mode and Gaussian smoothing across the viseme sequence."""
        if not raw_records or len(raw_records) < 3:
            return raw_records

        # 1. Smooth the continuous ratio curve
        ratios = np.array([r.mouth_open_ratio for r in raw_records], dtype=np.float32)
        pad = self.window_size // 2
        padded = np.pad(ratios, pad, mode="edge")
        
        weights = np.exp(-0.5 * (np.arange(self.window_size) - pad) ** 2)
        weights /= np.sum(weights)

        smoothed_ratios = np.convolve(padded, weights, mode="valid")

        # 2. Window Mode / Majority Vote Smoothing on discrete Visemes
        visemes = [r.viseme for r in raw_records]
        smoothed_visemes = []

        for i in range(len(raw_records)):
            start_i = max(0, i - pad)
            end_i = min(len(raw_records), i + pad + 1)
            window = visemes[start_i:end_i]
            
            # Most frequent state in window
            mode_viseme = Counter(window).most_common(1)[0][0]
            smoothed_visemes.append(mode_viseme)

        # 3. Construct clean smoothed records
        smoothed_records: List[LipSyncRecord] = []
        for i, raw_r in enumerate(raw_records):
            smoothed_records.append(
                LipSyncRecord(
                    frame_index=raw_r.frame_index,
                    timestamp=raw_r.timestamp,
                    mouth_open_ratio=float(smoothed_ratios[i]),
                    viseme=smoothed_visemes[i],
                )
            )

        return smoothed_records
