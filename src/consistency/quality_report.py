from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class IdentityQualityReport:

    total_frames: int = 0

    evaluated_frames: int = 0

    warning_frames: int = 0

    severe_drift_frames: int = 0

    retry_count: int = 0

    accepted_after_retry: int = 0

    minimum_score: float = 1.0

    maximum_score: float = 0.0

    average_score: float = 0.0

    score_sum: float = 0.0

    def add(
        self,
        score: float,
        warning: bool,
        severe: bool,
    ) -> None:

        self.evaluated_frames += 1

        self.score_sum += score

        self.minimum_score = min(
            self.minimum_score,
            score,
        )

        self.maximum_score = max(
            self.maximum_score,
            score,
        )

        if warning:
            self.warning_frames += 1

        if severe:
            self.severe_drift_frames += 1

        self.average_score = (
            self.score_sum
            / self.evaluated_frames
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

        with path.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                asdict(self),
                handle,
                indent=2,
            )
