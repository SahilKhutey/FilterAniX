"""Automated Regression Tests for Phase 2 Vision Engine on Synthetic Creator Fixture."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from tests.fixtures.generate_creator_video import generate_synthetic_creator_video
from src.vision.vision_pipeline import VisionEngine
from src.vision.models import FrameVisionData


@pytest.fixture(scope="module")
def synthetic_video_fixture(tmp_path_factory) -> Path:
    """Generates a dedicated synthetic creator video fixture with head, eyes, mouth, hand, and motion."""
    tmp_dir = tmp_path_factory.mktemp("synth_fixture")
    video_path = tmp_dir / "creator_test_video.mp4"
    generate_synthetic_creator_video(video_path, num_frames=30, width=640, height=360, fps=30)
    assert video_path.exists()
    return video_path


def test_vision_engine_synthetic_creator_detection(synthetic_video_fixture):
    """Verifies that the real Phase 2 Vision Engine reliably detects creator facial landmarks, blinks, and motion."""
    cap = cv2.VideoCapture(str(synthetic_video_fixture))
    assert cap.isOpened()

    engine = VisionEngine(enable_objects=False)

    total_frames = 0
    face_frames = 0
    blink_open_frames = 0
    blink_closed_frames = 0
    motion_active_frames = 0
    segmentation_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vision_data, annotated = engine.process_frame(
            rgb,
            frame_index=total_frames - 1,
            timestamp=(total_frames - 1) / 30.0,
            generate_annotated=True,
        )

        assert isinstance(vision_data, FrameVisionData)
        assert vision_data.width == 640
        assert vision_data.height == 360

        # Face & Landmark assertions
        if vision_data.faces and len(vision_data.faces) > 0:
            face_frames += 1
            face = vision_data.faces[0]
            assert face.landmark_count >= 468
            assert face.bbox is not None

            # Blinking / Eye Aspect Ratio dynamics
            ear = (face.left_eye_ear + face.right_eye_ear) / 2.0
            if ear > 0.15:
                blink_open_frames += 1
            else:
                blink_closed_frames += 1

        # Segmentation mask statistics
        if vision_data.person_mask and vision_data.person_mask.coverage > 0.0:
            segmentation_frames += 1

        # Optical flow motion
        if vision_data.motion and vision_data.motion.valid and vision_data.motion.mean_magnitude > 0.01:
            motion_active_frames += 1

    cap.release()

    assert total_frames == 30

    # Face detection rate should be high (> 50% threshold)
    face_detection_rate = face_frames / total_frames
    assert face_detection_rate >= 0.80

    # Blinking dynamics must report both open and closed eye frames
    assert blink_open_frames > 0
    assert blink_closed_frames > 0

    # Motion tracking must detect frame movements
    assert motion_active_frames > 0
