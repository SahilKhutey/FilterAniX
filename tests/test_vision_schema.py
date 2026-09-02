import pytest

from src.vision.schema import validate_vision_frame
from src.vision.types import VisionFrame
from src.vision.alignment import (
    validate_frame_sequence,
)


def make_frame(index: int = 0) -> dict:
    return VisionFrame(
        frame_index=index,
        timestamp=index / 30.0,
        width=640,
        height=480,
    ).to_dict()


def test_vision_frame_schema():
    frame = make_frame()
    validate_vision_frame(frame)


def test_frame_sequence():
    frames = [
        make_frame(0),
        make_frame(1),
        make_frame(2),
    ]
    validate_frame_sequence(frames)


def test_frame_sequence_detects_gap():
    frames = [
        make_frame(0),
        make_frame(1),
        make_frame(3),
    ]
    with pytest.raises(ValueError):
        validate_frame_sequence(frames)


def test_missing_field_is_rejected():
    frame = make_frame()
    del frame["face"]
    with pytest.raises(ValueError):
        validate_vision_frame(frame)


def test_invalid_face_detected_is_rejected():
    frame = make_frame()
    frame["face"]["detected"] = "yes"
    with pytest.raises(TypeError):
        validate_vision_frame(frame)
