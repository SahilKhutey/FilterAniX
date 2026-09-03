from __future__ import annotations

from typing import Optional

import numpy as np

from src.consistency.identity_bank import (
    IdentityReference,
    IdentityReferenceBank,
)


class IdentityReferenceSelector:

    def __init__(
        self,
        bank: IdentityReferenceBank,
    ):
        self.bank = bank

    def select(
        self,
        scene_id: int,
        current_frame_index: int = 0,
    ) -> Optional[np.ndarray]:

        reference = self.bank.best(
            scene_id=scene_id
        )

        if reference is None:
            return None

        return reference.image.copy()

    def add_reference(
        self,
        frame_index: int,
        scene_id: int,
        image_rgb: np.ndarray,
        identity_score: float = 1.0,
    ) -> None:

        self.bank.add(
            IdentityReference(
                frame_index=frame_index,
                image=image_rgb.copy(),
                face_crop=None,
                identity_score=identity_score,
                scene_id=scene_id,
            )
        )
