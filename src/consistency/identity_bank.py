from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class IdentityReference:
    frame_index: int

    image: np.ndarray

    face_crop: Optional[np.ndarray]

    identity_score: float = 1.0

    scene_id: int = 0

    metadata: dict = field(
        default_factory=dict
    )


class IdentityReferenceBank:

    def __init__(
        self,
        max_references: int = 8,
    ):

        self.max_references = max(
            1,
            int(max_references),
        )

        self.references: list[
            IdentityReference
        ] = []

    def add(
        self,
        reference: IdentityReference,
    ) -> None:

        self.references.append(
            reference
        )

        self.references.sort(
            key=lambda item: (
                item.identity_score
            ),
            reverse=True,
        )

        self.references = (
            self.references[
                : self.max_references
            ]
        )

    def clear_scene(
        self,
        scene_id: int,
    ) -> None:

        self.references = [
            item
            for item in self.references
            if item.scene_id == scene_id
        ]

    def best(
        self,
        scene_id: Optional[int] = None,
    ) -> Optional[IdentityReference]:

        candidates = self.references

        if scene_id is not None:

            scene_candidates = [
                item
                for item in candidates
                if item.scene_id == scene_id
            ]

            if scene_candidates:
                candidates = scene_candidates

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: (
                item.identity_score
            ),
        )

    def save(
        self,
        directory: str | Path,
    ) -> None:

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for reference in self.references:

            path = (
                directory
                / f"reference_{reference.frame_index:08d}.png"
            )

            cv2.imwrite(
                str(path),
                cv2.cvtColor(
                    reference.image,
                    cv2.COLOR_RGB2BGR,
                ),
            )
