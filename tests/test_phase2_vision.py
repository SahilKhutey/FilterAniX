"""Automated Tests for Phase 2 Vision Engine and Scene Understanding."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.vision.models import Landmark, BoundingBox, FaceData, PoseData, FrameVisionData
from src.vision.vision_pipeline import VisionEngine
from src.vision.face_engine import FaceEngine
from src.vision.motion_engine import MotionEngine


def test_vision_models_serialization():
    """Verifies that normalized vision data structures serialize cleanly to JSON."""
    frame_data = FrameVisionData(
        frame_index=1,
        timestamp=0.033,
        width=1920,
        height=1080,
        faces=[
            FaceData(
                face_id=0,
                landmarks=[Landmark(x=0.5, y=0.4, z=-0.05, visibility=1.0)],
                bbox=BoundingBox(x=0.4, y=0.3, width=0.2, height=0.25),
                landmark_count=1,
                mouth_opening=0.25,
                left_eye_ear=0.32,
                right_eye_ear=0.31,
            )
        ],
    )

    d = frame_data.to_dict()
    assert d["frame_index"] == 1
    assert d["width"] == 1920
    assert len(d["faces"]) == 1
    assert d["faces"][0]["mouth_opening"] == 0.25
    
    # Must be valid JSON stringifiable
    json_str = json.dumps(d)
    assert len(json_str) > 50


def test_face_engine_empty_image():
    """Verifies FaceEngine handles zero-face scenarios gracefully without crashing."""
    empty_img = np.zeros((300, 300, 3), dtype=np.uint8)
    engine = FaceEngine()
    faces = engine.process(empty_img)
    engine.close()
    assert isinstance(faces, list)
    assert len(faces) == 0


def test_motion_engine_translation():
    """Verifies optical flow motion engine accurately tracks translated frames."""
    engine = MotionEngine()

    frame1 = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(frame1, (100, 100), 20, (255, 255, 255), -1)

    frame2 = np.zeros((200, 200, 3), dtype=np.uint8)
    # Move circle 10 pixels to the right
    cv2.circle(frame2, (110, 100), 20, (255, 255, 255), -1)

    _, m1 = engine.process(frame1)
    assert m1.valid is False  # First frame has no motion history

    _, m2 = engine.process(frame2)
    assert m2.valid is True
    assert m2.mean_magnitude > 0.0
    assert m2.moving_pixel_ratio > 0.0


def test_full_vision_engine_pipeline(tmp_path):
    """Verifies the complete VisionEngine on a synthetic scene."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = [180, 190, 200]
    # Draw simple creator silhouette
    cv2.ellipse(img, (320, 200), (50, 70), 0, 0, 360, (220, 190, 170), -1)
    cv2.rectangle(img, (240, 270), (400, 480), (50, 60, 80), -1)

    engine = VisionEngine()
    data, annotated = engine.process_frame(img, frame_index=0, timestamp=0.0, generate_annotated=True)
    engine.close()

    assert data.width == 640
    assert data.height == 480
    assert annotated is not None
    assert annotated.shape == (480, 640, 3)
