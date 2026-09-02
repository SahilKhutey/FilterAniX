from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set


@dataclass
class ReferenceImage:
    name: str
    path: str
    tags: set[str]


class ReferenceBank:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        if not self.directory.exists():
            raise FileNotFoundError(self.directory)

        self.references = self._load()

    def _load(self) -> list[ReferenceImage]:
        result = []
        for path in sorted(self.directory.glob("*")):
            if path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                continue

            result.append(
                ReferenceImage(
                    name=path.stem,
                    path=str(path),
                    tags=set(path.stem.lower().split("_")),
                )
            )

        return result

    def select(
        self,
        *,
        expression: str | None = None,
        pose: str | None = None,
        profile: str | None = None,
    ) -> ReferenceImage | None:
        requested = set()

        for value in (
            expression,
            pose,
            profile,
        ):
            if value:
                requested.add(value.lower())

        if not requested:
            return self.references[0] if self.references else None

        best = None
        best_score = -1

        for reference in self.references:
            score = len(requested & reference.tags)
            if score > best_score:
                best = reference
                best_score = score

        return best
