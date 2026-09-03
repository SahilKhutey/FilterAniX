from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical.config import (
    MathematicalAnimeStyle,
)

from src.art.mathematical.color_field import (
    MathematicalColorField,
)


def make_test_frame(
    height: int = 64,
    width: int = 96,
) -> np.ndarray:

    y, x = np.mgrid[
        0:height,
        0:width,
    ]

    r = (
        60
        + 120 * x / max(1, width - 1)
    )

    g = (
        50
        + 140 * y / max(1, height - 1)
    )

    b = (
        70
        + 80
        * (
            x + y
        )
        / max(
            1,
            width + height - 2,
        )
    )

    frame = np.stack(
        [r, g, b],
        axis=-1,
    )

    return np.clip(
        frame,
        0,
        255,
    ).astype(
        np.uint8
    )


def test_engine_creates_output():

    engine = MathematicalColorField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape

    assert output.dtype == np.uint8


def test_every_pixel_is_calculated():

    engine = MathematicalColorField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape

    assert np.isfinite(
        output
    ).all()


def test_output_range():

    engine = MathematicalColorField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    assert output.min() >= 0

    assert output.max() <= 255


def test_transformation_changes_image():

    engine = MathematicalColorField()

    frame = make_test_frame()

    output = engine.render(
        frame
    )

    difference = np.abs(
        output.astype(np.int16)
        - frame.astype(np.int16)
    )

    changed_ratio = np.mean(
        np.any(
            difference > 2,
            axis=2,
        )
    )

    assert changed_ratio > 0.50


def test_constant_image_remains_valid():

    engine = MathematicalColorField()

    frame = np.full(
        (32, 32, 3),
        128,
        dtype=np.uint8,
    )

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape

    assert output.dtype == np.uint8

    assert np.isfinite(
        output
    ).all()


def test_float_input():

    engine = MathematicalColorField()

    frame = (
        make_test_frame()
        .astype(np.float32)
        / 255.0
    )

    output = engine.render(
        frame
    )

    assert output.dtype == np.uint8

    assert output.shape == frame.shape


def test_invalid_shape():

    engine = MathematicalColorField()

    frame = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):

        engine.render(frame)


def test_result_contains_intermediate_fields():

    engine = MathematicalColorField()

    frame = make_test_frame()

    result = engine.transform(
        frame
    )

    assert result.output_rgb.shape == frame.shape

    assert result.smoothed_rgb.shape == frame.shape

    assert result.palette_rgb.shape == frame.shape

    assert result.luminance.shape in (frame.shape, frame.shape[:2])

    assert (
        result.quantized_luminance.shape
        == result.luminance.shape
    )

    assert (
        result.saturation.shape
        == result.luminance.shape
    )


def test_custom_style():

    style = MathematicalAnimeStyle(
        palette_mix=0.90,
        saturation=1.20,
        color_levels=8,
        smooth_sigma=1.5,
    ).validated()

    engine = MathematicalColorField(
        style
    )

    output = engine.render(
        make_test_frame()
    )

    assert output.dtype == np.uint8
