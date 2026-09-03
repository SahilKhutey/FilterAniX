from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class KeyframeCache:

    def __init__(
        self,
        directory: str | Path,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.root = self.directory

    def path_for(
        self,
        frame_index: int,
    ) -> Path:

        return (
            self.directory
            / f"keyframe_{frame_index:08d}.png"
        )

    def path(
        self,
        frame_index: int,
    ) -> Path:
        return self.path_for(frame_index)

    def exists(
        self,
        frame_index: int,
    ) -> bool:

        return self.path_for(frame_index).exists()

    def save(
        self,
        frame_index: int,
        rgb: np.ndarray,
    ) -> Path:

        path = self.path_for(frame_index)

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        ok = cv2.imwrite(
            str(path),
            bgr,
        )

        if not ok:
            raise IOError(
                f"Failed to save keyframe: {path}"
            )

        return path

    def load(
        self,
        frame_index: int,
    ) -> np.ndarray:

        path = self.path_for(frame_index)

        if not path.exists():
            raise FileNotFoundError(path)

        bgr = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if bgr is None:
            raise IOError(
                f"Failed to read keyframe: {path}"
            )

        return cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )
