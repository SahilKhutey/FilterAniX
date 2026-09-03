from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FramePacket:
    frame_index: int
    timestamp_seconds: float
    fps: float

    rgb: np.ndarray

    width: int
    height: int

    vision: Any | None = None
    temporal: Any | None = None

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rgb, np.ndarray):
            raise TypeError("rgb must be numpy.ndarray")

        if self.rgb.ndim != 3:
            raise ValueError("rgb must be HxWx3")

        if self.rgb.shape[2] != 3:
            raise ValueError("rgb must contain 3 channels")

        if self.rgb.shape[1] != self.width:
            raise ValueError("width does not match rgb")

        if self.rgb.shape[0] != self.height:
            raise ValueError("height does not match rgb")

        if self.metadata is None:
            self.metadata = {}
