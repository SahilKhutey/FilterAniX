import numpy as np

from src.art.keyframe_cache import KeyframeCache


def test_keyframe_cache_roundtrip(tmp_path):

    cache = KeyframeCache(
        tmp_path / "keyframes"
    )

    image = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    image[10:20, 10:20] = 255

    cache.save(
        12,
        image,
    )

    assert cache.exists(12)

    loaded = cache.load(12)

    assert loaded.shape == image.shape
    assert np.array_equal(
        loaded,
        image,
    )
