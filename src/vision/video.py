"""Video reading and frame iteration utilities for Vision Engine."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def iter_video_frames(
    video_path: str | Path,
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yields (frame_index, timestamp_seconds, raw_bgr_frame) sequentially from video."""
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / fps
            yield index, timestamp, frame
            index += 1
    finally:
        capture.release()


def count_video_frames(video_path: str | Path) -> int:
    """Returns the total number of frames in a video container."""
    capture = cv2.VideoCapture(str(video_path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()


def count_jsonl_records(path: str | Path) -> int:
    """Returns the number of non-empty records in a JSONL file."""
    count = 0
    with open(str(path), "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count
