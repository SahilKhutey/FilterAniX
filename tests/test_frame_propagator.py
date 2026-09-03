import numpy as np

from src.art.frame_propagator import FramePropagator


def test_propagator_output_shape():
    propagator = FramePropagator(blend=0.20)

    source_a = np.zeros((64, 64, 3), dtype=np.uint8)
    source_b = np.zeros((64, 64, 3), dtype=np.uint8)
    # Add a slight pattern to calculate flow
    source_a[20:40, 20:40] = 200
    source_b[22:42, 22:42] = 200

    art = np.ones((64, 64, 3), dtype=np.uint8) * 128

    result = propagator.warp(source_a, source_b, art)

    assert result.shape == art.shape
    assert result.dtype == np.uint8

    blended = propagator.blend_frames(result, art)
    assert blended.shape == art.shape
    assert blended.dtype == np.uint8
