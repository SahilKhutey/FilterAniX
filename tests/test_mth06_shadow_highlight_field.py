import numpy as np
import pytest

from src.art.mathematical import (
    MathematicalAnimeStyle,
    MathematicalShadowHighlightField,
)


def make_test_image(
    height: int = 128,
    width: int = 128,
) -> np.ndarray:

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Mid-tone background.
    image[:] = (
        150,
        120,
        110,
    )

    # Dark region.
    image[
        10:70,
        10:70
    ] = (
        35,
        30,
        40,
    )

    # Bright region.
    image[
        60:118,
        60:118
    ] = (
        245,
        220,
        185,
    )

    # Transition region.
    for y in range(40, 100):
        value = int(
            60
            + (
                y - 40
            )
            * 3
        )

        value = min(
            value,
            220,
        )

        image[y, 40:90] = (
            value,
            int(value * 0.85),
            int(value * 0.75),
        )

    return image


@pytest.fixture
def engine():

    style = (
        MathematicalAnimeStyle
        .creator_anime()
    )

    return MathematicalShadowHighlightField(
        style
    )


def test_output_shape(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert (
        result.output_rgb.shape
        == image.shape
    )


def test_output_dtype(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert (
        result.output_rgb.dtype
        == np.uint8
    )


def test_output_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert (
        result.output_rgb.min()
        >= 0
    )

    assert (
        result.output_rgb.max()
        <= 255
    )


def test_illumination_shapes(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    expected = image.shape[:2]

    assert (
        result.small_illumination.shape
        == expected
    )

    assert (
        result.medium_illumination.shape
        == expected
    )

    assert (
        result.large_illumination.shape
        == expected
    )

    assert (
        result.illumination_field.shape
        == expected
    )


def test_shadow_probability_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.shadow_probability
        >= 0.0
    )

    assert np.all(
        result.shadow_probability
        <= 1.0
    )


def test_highlight_probability_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.highlight_probability
        >= 0.0
    )

    assert np.all(
        result.highlight_probability
        <= 1.0
    )


def test_shadow_field_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.shadow_field
        >= 0.0
    )

    assert np.all(
        result.shadow_field
        <= 1.0
    )


def test_highlight_field_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.highlight_field
        >= 0.0
    )

    assert np.all(
        result.highlight_field
        <= 1.0
    )


def test_target_luminance_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.target_luminance
        >= 0.0
    )

    assert np.all(
        result.target_luminance
        <= 1.0
    )


def test_shadow_region_detected(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    dark_region = (
        result.shadow_field[
            20:60,
            20:60
        ].mean()
    )

    bright_region = (
        result.shadow_field[
            70:110,
            70:110
        ].mean()
    )

    assert dark_region > bright_region


def test_highlight_region_detected(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    bright_region = (
        result.highlight_field[
            70:110,
            70:110
        ].mean()
    )

    dark_region = (
        result.highlight_field[
            20:60,
            20:60
        ].mean()
    )

    assert bright_region > dark_region


def test_output_changes(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    difference = np.abs(
        result.output_rgb.astype(
            np.int16
        )
        - image.astype(
            np.int16
        )
    )

    changed_ratio = np.mean(
        difference > 2
    )

    assert changed_ratio > 0.05


def test_constant_image(engine):

    image = np.full(
        (64, 64, 3),
        128,
        dtype=np.uint8,
    )

    result = engine.transform(
        image
    )

    assert (
        result.output_rgb.shape
        == image.shape
    )

    assert np.allclose(
        result.illumination_field,
        128 / 255.0,
        atol=0.01,
    )


def test_float_input(engine):

    image = (
        make_test_image()
        .astype(np.float32)
        / 255.0
    )

    result = engine.transform(
        image
    )

    assert (
        result.output_rgb.dtype
        == np.uint8
    )


def test_invalid_channels(engine):

    image = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError
    ):
        engine.transform(
            image
        )


def test_invalid_type(engine):

    with pytest.raises(
        TypeError
    ):
        engine.transform(
            "invalid"
        )
