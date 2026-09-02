from __future__ import annotations

from typing import Iterable


def validate_frame_sequence(
    frames: Iterable[dict],
) -> None:
    expected = 0

    for frame in frames:
        index = frame["frame_index"]

        if index != expected:
            raise ValueError(
                "Vision frame sequence broken: "
                f"expected {expected}, got {index}"
            )

        expected += 1
