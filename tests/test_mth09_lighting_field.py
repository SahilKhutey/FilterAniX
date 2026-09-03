from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical import (
    GeometryBox,
    GeometryObservation,
    MathematicalAnimeStyle,
    MathematicalLightingField,
)


def make_image() -> np.ndarray:

    height = 128
    width = 128

    y, x = np.mgrid[
        0:height,
        0:width,
    ]

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[..., 0] = np.clip(
        60 + x * 1.2,
        0,
        255,
    )

    image[..., 1] = np.clip(
        50 + y * 1.1,
        0,
        255,
    )

    image[..., 2] = 70

    return image


def make_observation() -> GeometryObservation:

    return GeometryObservation(
        width=128,
        height=128,

        face_box=GeometryBox(
            x0=35,
            y0=20,
            x1=95,
            y1=100,
            confidence=1.0,
        ),

        face_landmarks=[],
        pose_landmarks=[],
        hand_landmarks=[],

        person_mask=np.ones(
            (128, 128),
            dtype=np.float32,
        ),
    )


@pytest.fixture
def engine():

    return MathematicalLightingField(
        MathematicalAnimeStyle.creator_anime()
    )


def test_output_shape(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.output_rgb.shape == (
        128,
        128,
        3,
    )


def test_output_dtype(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.output_rgb.dtype == np.uint8


def test_output_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.output_rgb.min() >= 0
    assert result.output_rgb.max() <= 255


def test_luminance_shape(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.luminance.shape == (
        128,
        128,
    )


def test_shadow_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.shadow_field.min() >= 0
    assert result.shadow_field.max() <= 1


def test_highlight_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.highlight_field.min() >= 0
    assert result.highlight_field.max() <= 1


def test_midtone_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.midtone_field.min() >= 0
    assert result.midtone_field.max() <= 1


def test_local_light_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.local_light_field.min() >= 0
    assert result.local_light_field.max() <= 1


def test_warm_light_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.warm_light_field.min() >= 0
    assert result.warm_light_field.max() <= 1


def test_face_protection_detected(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.face_protection_field.max() > 0.5


def test_highlight_protection_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert (
        result.highlight_protection_field.min()
        >= 0
    )

    assert (
        result.highlight_protection_field.max()
        <= 1
    )


def test_contributions_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert (
        result.shadow_contribution.min()
        >= 0
    )

    assert (
        result.shadow_contribution.max()
        <= 1
    )

    assert (
        result.key_light_contribution.min()
        >= 0
    )

    assert (
        result.key_light_contribution.max()
        <= 1
    )


def test_final_light_field_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.final_light_field.min() >= 0
    assert result.final_light_field.max() <= 1


def test_float_input_supported(engine):

    image = (
        make_image()
        .astype(np.float32)
        / 255.0
    )

    result = engine.transform(
        image,
        make_observation(),
    )

    assert result.output_rgb.dtype == np.uint8


def test_invalid_frame_raises(engine):

    with pytest.raises(ValueError):

        engine.transform(
            np.zeros(
                (128, 128),
                dtype=np.uint8,
            ),
            make_observation(),
        )


def test_resolution_mismatch_raises(engine):

    observation = make_observation()

    observation.width = 64

    with pytest.raises(ValueError):

        engine.transform(
            make_image(),
            observation,
        )


def test_constant_frame_is_stable(engine):

    image = np.full(
        (128, 128, 3),
        128,
        dtype=np.uint8,
    )

    result = engine.transform(
        image,
        make_observation(),
    )

    assert result.output_rgb.shape == image.shape

    assert np.isfinite(
        result.output_rgb
    ).all()
