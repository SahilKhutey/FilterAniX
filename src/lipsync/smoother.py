from __future__ import annotations

from collections import Counter
from typing import List, Union
import numpy as np

from .timeline import VisemeFrame
from .analyzer import LipSyncRecord


def smooth_timeline(
    frames: List[Union[VisemeFrame, LipSyncRecord]],
    window: int = 3,
) -> List[Union[VisemeFrame, LipSyncRecord]]:
    """Applies majority vote smoothing over sliding window to eliminate detector flicker."""
    if not frames:
        return []

    result = []
    half = window // 2

    is_viseme_frame = isinstance(frames[0], VisemeFrame)

    for i, current in enumerate(frames):
        start = max(0, i - half)
        end = min(len(frames), i + half + 1)

        if is_viseme_frame:
            states = [frames[j].state for j in range(start, end)]
            majority_state = Counter(states).most_common(1)[0][0]
            result.append(
                VisemeFrame(
                    frame_index=current.frame_index,
                    timestamp=current.timestamp,
                    mouth_open=current.mouth_open,
                    state=majority_state,
                )
            )
        else:
            states = [frames[j].viseme for j in range(start, end)]
            majority_state = Counter(states).most_common(1)[0][0]
            result.append(
                LipSyncRecord(
                    frame_index=current.frame_index,
                    timestamp=current.timestamp,
                    mouth_open_ratio=current.mouth_open_ratio,
                    viseme=majority_state,
                )
            )

    return result


class LipSyncSmoother:
    """Smooths raw frame-by-frame viseme timeline using rolling window filtering."""

    def __init__(self, window_size: int = 5):
        self.window_size = max(3, window_size if window_size % 2 != 0 else window_size + 1)

    def smooth_timeline(self, raw_records: List[LipSyncRecord]) -> List[LipSyncRecord]:
        return smooth_timeline(raw_records, window=self.window_size)
