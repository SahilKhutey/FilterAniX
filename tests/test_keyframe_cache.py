import numpy as np

from src.art.keyframe_cache import KeyframeCache


def test_keyframe_cache(tmp_path):
    cache = KeyframeCache(tmp_path)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[10:20, 10:20] = [255, 128, 64]

    cache.save(10, frame)

    assert cache.exists(10)
    assert not cache.exists(11)

    loaded = cache.load(10)

    assert loaded.shape == frame.shape
    assert np.allclose(loaded, frame, atol=2)
