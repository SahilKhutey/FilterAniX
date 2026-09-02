from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class VisemeFrame:
    frame_index: int
    timestamp: float
    mouth_open: float
    state: str

    def to_dict(self):
        return asdict(self)


@dataclass
class LipSyncTimeline:
    fps: float
    frames: List[VisemeFrame]

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for frame in self.frames:
                f.write(json.dumps(frame.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str, fps: float):
        frames = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                data = json.loads(line)

                frames.append(
                    VisemeFrame(
                        frame_index=int(data["frame_index"]),
                        timestamp=float(data["timestamp"]),
                        mouth_open=float(data["mouth_open"]),
                        state=str(data["state"]),
                    )
                )

        return cls(
            fps=fps,
            frames=frames,
        )
