import numpy as np

from src.consistency.identity_features import (
    extract_identity_features,
)

from src.consistency.identity_metric import (
    IdentityMetricEngine,
)


def test_identical_images_score_high():

    image = np.zeros(
        (128, 128, 3),
        dtype=np.uint8,
    )

    image[30:90, 30:90] = 180

    features = extract_identity_features(
        image
    )

    engine = IdentityMetricEngine()

    result = engine.compare(
        features,
        features,
    )

    assert result.overall > 0.90
    assert not result.warning
