from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class RenderMetrics:

    total_frames: int = 0

    keyframes: int = 0

    propagated_frames: int = 0

    diffusion_frames: int = 0

    fallback_frames: int = 0

    scene_cuts: int = 0

    render_seconds: float = 0.0

    @property
    def keyframe_ratio(self) -> float:

        if self.total_frames == 0:
            return 0.0

        return (
            self.keyframes
            / self.total_frames
        )

    @property
    def propagation_ratio(self) -> float:

        if self.total_frames == 0:
            return 0.0

        return (
            self.propagated_frames
            / self.total_frames
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = asdict(self)

        data["keyframe_ratio"] = (
            self.keyframe_ratio
        )

        data["propagation_ratio"] = (
            self.propagation_ratio
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                data,
                handle,
                indent=2,
            )
