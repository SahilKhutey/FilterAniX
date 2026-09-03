from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical.tone_field import (
    MathematicalToneField,
)


def make_gradient(
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

    g = yy

    b = (
        0.25
        + 0.50 * xx
        + 0.25 * yy
    )

    frame = np.stack(
        [r, g, b],
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

    engine = MathematicalToneField()

    frame = make_gradient()

    output = engine.render(
        frame
    )

    assert output.shape == frame.shape


def test_output_dtype():

    engine = MathematicalToneField()

    frame = make_gradient()

    output = engine.render(
        frame
    )

    assert output.dtype == np.uint8


def test_output_range():

    engine = MathematicalToneField()

    frame = make_gradient()

    output = engine.render(
        frame
    )

    assert output.min() >= 0
    assert output.max() <= 255


def test_tone_transformation_changes_image():

    engine = MathematicalToneField()

    frame = make_gradient()

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
            axis=2,
        )
    )

    assert changed_ratio > 0.20


def test_result_contains_all_fields():

    engine = MathematicalToneField()

    result = engine.transform(
        make_gradient()
    )

    assert result.output_rgb.ndim == 3

    assert result.luminance.ndim == 2

    assert result.local_luminance.ndim == 2

    assert result.local_detail.ndim == 2

    assert result.normalized_detail.ndim == 2

    assert result.quantized_luminance.ndim == 2

    assert result.shadow_mask.ndim == 2

    assert result.highlight_mask.ndim == 2

    assert result.target_luminance.ndim == 2


def test_shadow_mask_range():

    engine = MathematicalToneField()

    result = engine.transform(
        make_gradient()
    )

    assert result.shadow_mask.min() >= 0.0

    assert result.shadow_mask.max() <= 1.0


def test_highlight_mask_range():

    engine = MathematicalToneField()

    result = engine.transform(
        make_gradient()
    )

    assert result.highlight_mask.min() >= 0.0

    assert result.highlight_mask.max() <= 1.0


def test_target_luminance_range():

    engine = MathematicalToneField()

    result = engine.transform(
        make_gradient()
    )

    assert (
        result.target_luminance.min()
        >= 0.0
    )

    assert (
        result.target_luminance.max()
        <= 1.0
    )


def test_constant_frame():

    engine = MathematicalToneField()

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

    engine = MathematicalToneField()

    frame = (
        make_gradient()
        .astype(np.float32)
        / 255.0
    )

    output = engine.render(
        frame
    )

    assert output.dtype == np.uint8


def test_invalid_input():

    engine = MathematicalToneField()

    invalid = np.zeros(
        (32, 32),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError
    ):
        engine.render(
            invalid
        )
