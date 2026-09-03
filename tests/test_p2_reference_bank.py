import numpy as np

from src.consistency.identity_bank import (
    IdentityReference,
    IdentityReferenceBank,
)


def test_reference_bank():

    bank = IdentityReferenceBank(
        max_references=2
    )

    image = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    bank.add(
        IdentityReference(
            frame_index=0,
            image=image,
            face_crop=None,
            identity_score=0.90,
            scene_id=0,
        )
    )

    bank.add(
        IdentityReference(
            frame_index=1,
            image=image,
            face_crop=None,
            identity_score=0.95,
            scene_id=0,
        )
    )

    best = bank.best(
        scene_id=0
    )

    assert best is not None
    assert best.frame_index == 1
