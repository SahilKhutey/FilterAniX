import numpy as np
import pytest

from src.art.mathematical import (
    GeometryBox,
    GeometryObservation,
    GeometryPoint,
    MathematicalGeometryField,
)


def make_image(
    height=128,
    width=128,
):

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[:] = (
        180,
        150,
        140,
    )

    image[20:105, 35:95] = (
        100,
        80,
        90,
    )

    return image


def make_observation(
    height=128,
    width=128,
):

    face_box = GeometryBox(
        x0=45,
        y0=20,
        x1=85,
        y1=60,
        confidence=1.0,
    )

    face_landmarks = [
        GeometryPoint(
            55,
            35,
            1.0,
        ),
        GeometryPoint(
            75,
            35,
            1.0,
        ),
        GeometryPoint(
            65,
            42,
            1.0,
        ),
        GeometryPoint(
            65,
            52,
            1.0,
        ),
    ]

    pose_landmarks = [
        GeometryPoint(
            50,
            65,
            1.0,
        ),
        GeometryPoint(
            80,
            65,
            1.0,
        ),
        GeometryPoint(
            45,
            95,
            1.0,
        ),
        GeometryPoint(
            85,
            95,
            1.0,
        ),
    ]

    hand_landmarks = [
        GeometryPoint(
            35,
            90,
            1.0,
        ),
        GeometryPoint(
            100,
            90,
            1.0,
        ),
    ]

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    mask[20:110, 30:100] = 1.0

    return GeometryObservation(
        width=width,
        height=height,
        face_box=face_box,
        face_landmarks=face_landmarks,
        pose_landmarks=pose_landmarks,
        hand_landmarks=hand_landmarks,
        person_mask=mask,
    )


@pytest.fixture
def engine():

    return MathematicalGeometryField()


def test_output_shape(engine):

    image = make_image()
    observation = make_observation()

    result = engine.transform(
        image,
        observation,
    )

    assert (
        result.output_rgb.shape
        == image.shape
    )


def test_output_dtype(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert (
        result.output_rgb.dtype
        == np.uint8
    )


def test_field_shapes(engine):

    image = make_image()

    result = engine.transform(
        image,
        make_observation(),
    )

    expected = image.shape[:2]

    assert result.face_field.shape == expected
    assert result.face_landmark_field.shape == expected
    assert result.pose_field.shape == expected
    assert result.hand_field.shape == expected
    assert result.person_field.shape == expected
    assert result.character_field.shape == expected
    assert result.background_field.shape == expected
    assert result.face_importance.shape == expected
    assert result.structural_importance.shape == expected
    assert result.detail_preservation.shape == expected
    assert result.simplification_field.shape == expected


def test_fields_range(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    fields = [
        result.face_field,
        result.face_landmark_field,
        result.pose_field,
        result.hand_field,
        result.person_field,
        result.character_field,
        result.background_field,
        result.face_importance,
        result.structural_importance,
        result.detail_preservation,
        result.simplification_field,
    ]

    for field in fields:

        assert np.all(
            field >= 0.0
        )

        assert np.all(
            field <= 1.0
        )


def test_face_detected(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    face_value = (
        result.face_field[
            35:55,
            50:80
        ].mean()
    )

    background_value = (
        result.face_field[
            100:120,
            5:25
        ].mean()
    )

    assert face_value > background_value


def test_character_detected(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    character = (
        result.character_field[
            30:100,
            35:95
        ].mean()
    )

    background = (
        result.character_field[
            0:15,
            0:15
        ].mean()
    )

    assert character > background


def test_background_complement(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert np.allclose(
        result.background_field
        + result.character_field,
        1.0,
        atol=0.10,
    )


def test_face_importance(engine):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    value = (
        result.face_importance[
            30:55,
            50:80
        ].mean()
    )

    assert value > 0.2


def test_float_input(engine):

    image = (
        make_image()
        .astype(np.float32)
        / 255.0
    )

    result = engine.transform(
        image,
        make_observation(),
    )

    assert (
        result.output_rgb.dtype
        == np.uint8
    )


def test_resolution_mismatch(engine):

    image = make_image()

    observation = make_observation(
        height=64,
        width=64,
    )

    with pytest.raises(
        ValueError
    ):

        engine.transform(
            image,
            observation,
        )


def test_invalid_frame(engine):

    with pytest.raises(
        ValueError
    ):

        engine.transform(
            np.zeros(
                (64, 64),
                dtype=np.uint8,
            ),
            make_observation(
                height=64,
                width=64,
            ),
        )
