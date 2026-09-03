from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical import (
    GeometryBox,
    GeometryObservation,
    GeometryPoint,
    MathematicalAnimeStyle,
    MathematicalFaceField,
)


def make_image() -> np.ndarray:
    height = 160
    width = 160

    y, x = np.mgrid[0:height, 0:width]

    image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    image[..., 0] = np.clip(
        120 + x * 0.25,
        0,
        255,
    )

    image[..., 1] = np.clip(
        100 + y * 0.20,
        0,
        255,
    )

    image[..., 2] = 90

    return image


def make_observation() -> GeometryObservation:

    landmarks = [
        GeometryPoint(
            x=80.0,
            y=70.0,
            confidence=1.0,
        )
        for _ in range(468)
    ]

    # MediaPipe-style positions.

    landmarks[33] = GeometryPoint(
        x=60.0,
        y=65.0,
        confidence=1.0,
    )

    landmarks[263] = GeometryPoint(
        x=100.0,
        y=65.0,
        confidence=1.0,
    )

    landmarks[1] = GeometryPoint(
        x=80.0,
        y=78.0,
        confidence=1.0,
    )

    landmarks[168] = GeometryPoint(
        x=80.0,
        y=70.0,
        confidence=1.0,
    )

    landmarks[61] = GeometryPoint(
        x=67.0,
        y=95.0,
        confidence=1.0,
    )

    landmarks[291] = GeometryPoint(
        x=93.0,
        y=95.0,
        confidence=1.0,
    )

    landmarks[13] = GeometryPoint(
        x=80.0,
        y=92.0,
        confidence=1.0,
    )

    landmarks[14] = GeometryPoint(
        x=80.0,
        y=98.0,
        confidence=1.0,
    )

    landmarks[10] = GeometryPoint(
        x=80.0,
        y=42.0,
        confidence=1.0,
    )

    landmarks[152] = GeometryPoint(
        x=80.0,
        y=120.0,
        confidence=1.0,
    )

    person_mask = np.zeros(
        (160, 160),
        dtype=np.float32,
    )

    cv_y, cv_x = np.ogrid[:160, :160]

    person_mask[
        (cv_x - 80) ** 2 +
        (cv_y - 80) ** 2 < 60 ** 2
    ] = 1.0

    return GeometryObservation(
        width=160,
        height=160,

        face_box=GeometryBox(
            x0=40,
            y0=40,
            x1=120,
            y1=125,
            confidence=1.0,
        ),

        face_landmarks=landmarks,

        pose_landmarks=[],
        hand_landmarks=[],

        person_mask=person_mask,
    )


@pytest.fixture
def engine() -> MathematicalFaceField:

    return MathematicalFaceField(
        MathematicalAnimeStyle.creator_anime()
    )


def test_output_shape(
    engine: MathematicalFaceField,
):

    image = make_image()
    observation = make_observation()

    result = engine.transform(
        image,
        observation,
    )

    assert result.output_rgb.shape == image.shape


def test_output_dtype(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.output_rgb.dtype == np.uint8


def test_output_range(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.output_rgb.min() >= 0
    assert result.output_rgb.max() <= 255


def test_all_fields_have_correct_shape(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    expected = (160, 160)

    fields = [
        result.face_field,
        result.eye_field,
        result.nose_field,
        result.mouth_field,
        result.central_feature_field,
        result.facial_geometry_field,
        result.face_importance,
        result.detail_preservation,
        result.smoothing_field,
        result.eye_emphasis,
        result.mouth_emphasis,
        result.nose_emphasis,
    ]

    for field in fields:
        assert field.shape == expected


def test_fields_are_normalized(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    fields = [
        result.face_field,
        result.eye_field,
        result.nose_field,
        result.mouth_field,
        result.central_feature_field,
        result.facial_geometry_field,
        result.face_importance,
        result.detail_preservation,
        result.smoothing_field,
        result.eye_emphasis,
        result.mouth_emphasis,
        result.nose_emphasis,
    ]

    for field in fields:
        assert field.min() >= 0.0
        assert field.max() <= 1.0


def test_face_field_detected(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.face_field.mean() > 0.05


def test_eye_field_detected(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.eye_field.max() > 0.5


def test_nose_field_detected(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.nose_field.max() > 0.5


def test_mouth_field_detected(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.mouth_field.max() > 0.5


def test_feature_field_is_not_empty(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.central_feature_field.max() > 0.5


def test_face_importance_detected(
    engine: MathematicalFaceField,
):

    result = engine.transform(
        make_image(),
        make_observation(),
    )

    assert result.face_importance.mean() > 0.02


def test_float_input_supported(
    engine: MathematicalFaceField,
):

    image = make_image().astype(
        np.float32
    ) / 255.0

    result = engine.transform(
        image,
        make_observation(),
    )

    assert result.output_rgb.dtype == np.uint8


def test_resolution_mismatch_raises(
    engine: MathematicalFaceField,
):

    observation = make_observation()

    observation.width = 128

    with pytest.raises(ValueError):
        engine.transform(
            make_image(),
            observation,
        )


def test_invalid_frame_raises(
    engine: MathematicalFaceField,
):

    with pytest.raises(ValueError):
        engine.transform(
            np.zeros(
                (160, 160),
                dtype=np.uint8,
            ),
            make_observation(),
        )


def test_missing_landmarks_do_not_crash(
    engine: MathematicalFaceField,
):

    observation = make_observation()

    observation.face_landmarks = []

    result = engine.transform(
        make_image(),
        observation,
    )

    assert result.output_rgb.shape == (
        160,
        160,
        3,
    )

    assert result.eye_field.max() == 0.0
    assert result.nose_field.max() == 0.0
    assert result.mouth_field.max() == 0.0
