import numpy as np
import pytest

from src.art.mathematical import (
    MathematicalAnimeStyle,
    MathematicalEdgeField,
)


def make_test_image(
    height: int = 128,
    width: int = 128,
) -> np.ndarray:

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    # Background
    image[:] = (
        210,
        170,
        145,
    )

    # Large dark region
    image[20:100, 25:105] = (
        50,
        40,
        55,
    )

    # Bright region
    image[40:80, 45:85] = (
        245,
        220,
        180,
    )

    # Structural lines
    image[10:115, 62:66] = (
        20,
        20,
        25,
    )

    image[62:66, 10:115] = (
        20,
        20,
        25,
    )

    return image


@pytest.fixture
def engine():

    style = MathematicalAnimeStyle.creator_anime()

    return MathematicalEdgeField(
        style
    )


def test_output_shape(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert result.output_rgb.shape == image.shape


def test_output_dtype(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert result.output_rgb.dtype == np.uint8


def test_output_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert result.output_rgb.min() >= 0
    assert result.output_rgb.max() <= 255


def test_luminance_shape(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert result.luminance.shape == image.shape[:2]


def test_gradient_shapes(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    expected = image.shape[:2]

    assert result.gradient_x.shape == expected
    assert result.gradient_y.shape == expected
    assert result.gradient_magnitude.shape == expected
    assert result.gradient_orientation.shape == expected


def test_laplacian_shape(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert result.laplacian.shape == image.shape[:2]


def test_multiscale_shapes(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    expected = image.shape[:2]

    assert result.small_scale_response.shape == expected
    assert result.medium_scale_response.shape == expected
    assert result.large_scale_response.shape == expected
    assert result.multiscale_response.shape == expected


def test_edge_probability_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.edge_probability >= 0.0
    )

    assert np.all(
        result.edge_probability <= 1.0
    )


def test_line_strength_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.line_strength >= 0.0
    )

    assert np.all(
        result.line_strength <= 1.0
    )


def test_line_field_range(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert np.all(
        result.line_field >= 0.0
    )

    assert np.all(
        result.line_field <= 1.0
    )


def test_edges_exist(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    assert (
        float(
            np.mean(
                result.edge_probability
            )
        )
        > 0.0
    )


def test_image_changes(engine):

    image = make_test_image()

    result = engine.transform(
        image
    )

    difference = np.abs(
        result.output_rgb.astype(np.int16)
        - image.astype(np.int16)
    )

    changed_ratio = np.mean(
        difference > 2
    )

    assert changed_ratio > 0.10


def test_constant_image(engine):

    image = np.full(
        (64, 64, 3),
        128,
        dtype=np.uint8,
    )

    result = engine.transform(
        image
    )

    assert result.output_rgb.shape == image.shape

    assert np.all(
        result.gradient_magnitude
        < 1e-5
    )


def test_float_input(engine):

    image = make_test_image().astype(
        np.float32
    ) / 255.0

    result = engine.transform(
        image
    )

    assert result.output_rgb.dtype == np.uint8


def test_invalid_channels(engine):

    image = np.zeros(
        (64, 64),
        dtype=np.uint8,
    )

    with pytest.raises(ValueError):

        engine.transform(
            image
        )


def test_invalid_type(engine):

    with pytest.raises(TypeError):

        engine.transform(
            "invalid"
        )
