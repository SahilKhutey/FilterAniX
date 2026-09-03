import numpy as np

from src.art.frame_propagator import FramePropagator


def test_propagator_shape():

    propagator = FramePropagator()

    previous_source = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    current_source = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    previous_art = np.zeros(
        (64, 64, 3),
        dtype=np.uint8,
    )

    output = propagator.warp(
        previous_source,
        current_source,
        previous_art,
    )

    assert output.shape == (
        64,
        64,
        3,
    )
