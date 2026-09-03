from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical.palette_field import (
    MathematicalPaletteField,
)


def make_test_frame(
    height: int = 64,
    width: int = 96,
) -> np.ndarray:

    x = np.linspace(
        0.0,
        1.0,
        width,
        dtype=np.float32,
    )

    y = np.linspace(
        0.0,
        1.0,
        height,
        dtype=np.float32,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    r = xx

    g = (
        0.30
        + 0.50 * yy
    )

    b = (
        0.20
        + 0.50 * (1.0 - xx)
    )

    frame = np.stack(
        [
            r,
            g,
            b,
        ],
        axis=-1,
    )

    return (
        np.clip(
            frame,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )


def test_output_shape():

    engine = MathematicalPaletteField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape


def test_output_dtype():

    engine = MathematicalPaletteField()

    output = engine.render(
        make_test_frame()
    )

    assert output.dtype == np.uint8


def test_output_range():

    engine = MathematicalPaletteField()

    output = engine.render(
        make_test_frame()
    )

    assert output.min() >= 0
    assert output.max() <= 255


def test_every_pixel_has_palette_weights():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    h, w, _ = result.input_rgb.shape

    assert result.weights.shape == (
        h,
        w,
        len(engine.style.palette),
    )


def test_weights_sum_to_one():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    sums = np.sum(
        result.weights,
        axis=-1,
    )

    assert np.allclose(
        sums,
        1.0,
        atol=1e-5,
    )


def test_palette_field_shape():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    assert (
        result.palette_rgb.shape
        == result.input_rgb.shape
    )


def test_confidence_range():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    assert result.confidence.min() >= 0.0

    assert result.confidence.max() <= 1.0


def test_entropy_range():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    assert result.palette_entropy.min() >= 0.0

    assert result.palette_entropy.max() <= 1.0


def test_dominant_palette_shape():

    engine = MathematicalPaletteField()

    result = engine.transform(
        make_test_frame()
    )

    h, w, _ = result.input_rgb.shape

    assert result.dominant_index.shape == (
        h,
        w,
    )


def test_palette_changes_image():

    engine = MathematicalPaletteField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    difference = np.abs(
        output.astype(
            np.int16
        )
        - frame.astype(
            np.int16
        )
    )

    changed_ratio = np.mean(
        np.any(
            difference > 2,
            axis=-1,
        )
    )

    assert changed_ratio > 0.20


def test_constant_image():

    engine = MathematicalPaletteField()

    frame = np.full(
        (32, 32, 3),
        128,
        dtype=np.uint8,
    )

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape

    assert np.isfinite(
        output
    ).all()


def test_float_input():

    engine = MathematicalPaletteField()

    frame = (
        make_test_frame()
        .astype(
            np.float32
        )
        / 255.0
    )

    output = engine.render(
        frame
    )

    assert output.dtype == np.uint8


def test_invalid_input():

    engine = MathematicalPaletteField()

    frame = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError
    ):
        engine.render(
            frame
        )
