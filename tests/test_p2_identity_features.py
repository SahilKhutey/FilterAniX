import numpy as np

from src.consistency.identity_features import (
    extract_identity_features,
)


def test_identity_features():

    image = np.zeros(
        (128, 128, 3),
        dtype=np.uint8,
    )

    image[32:96, 32:96] = 255

    features = extract_identity_features(
        image
    )

    assert features.face_gray.shape == (
        128,
        128,
    )

    assert features.face_histogram.size > 0

    assert features.color_histogram.size > 0
