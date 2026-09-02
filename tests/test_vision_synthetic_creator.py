"""Automated Regression Tests for Phase 2 Vision Engine on Synthetic Creator Fixture."""
from pathlib import Path
import json
import cv2
import numpy as np
import pytest

from tests.fixtures import ensure_creator_video
from src.vision.vision_pipeline import VisionEngine
from src.vision.models import FrameVisionData
from src.vision.video import iter_video_frames, count_video_frames, count_jsonl_records
from analyze_video import analyze_video


def test_synthetic_creator_has_meaningful_vision():
    """Verifies that the real Phase 2 Vision Engine reliably detects creator facial landmarks, blinks, and motion."""
    video = ensure_creator_video()
    assert video.exists()

    engine = VisionEngine(enable_objects=False)

    total_frames = 0
    face_frames = 0
    blink_open_frames = 0
    blink_closed_frames = 0
    motion_active_frames = 0
    segmentation_frames = 0

    for index, timestamp, frame in iter_video_frames(video):
        total_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        vision_data, annotated = engine.process_frame(
            rgb,
            frame_index=index,
            timestamp=timestamp,
            generate_annotated=False,
        )

        assert isinstance(vision_data, FrameVisionData)
        assert vision_data.width == 640
        assert vision_data.height == 480

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
        if vision_data.motion and vision_data.motion.valid and vision_data.motion.mean_magnitude > 0.005:
            motion_active_frames += 1

    engine.face_engine.close()
    engine.pose_engine.close()
    engine.hand_engine.close()

    assert total_frames >= 150

    # Face detection rate must meet behavioral minimums
    face_detection_rate = face_frames / total_frames
    assert face_detection_rate >= 0.50

    # Blinking dynamics must report both open and closed eye frames
    assert blink_open_frames > 0
    assert blink_closed_frames > 0

    # Motion tracking must detect frame movements
    assert motion_active_frames > 0


def test_vision_frame_alignment_and_jsonl(tmp_path):
    """Verifies that analyze_video writes valid JSONL records matching frame count exactly."""
    video = ensure_creator_video()
    out_dir = tmp_path / "analysis_run"
    
    analyze_video(str(video), output_dir=str(out_dir), max_frames=30)

    jsonl_path = out_dir / "vision.jsonl"
    summary_path = out_dir / "summary.json"
    annotated_path = out_dir / "annotated.mp4"

    assert jsonl_path.exists()
    assert summary_path.exists()
    assert annotated_path.exists()

    # Verify frame alignment
    record_count = count_jsonl_records(jsonl_path)
    assert record_count == 30

    # Verify JSONL lines schema
    lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 30

    for i, line in enumerate(lines):
        record = json.loads(line)
        assert record["frame_index"] == i
        assert record["width"] == 640
        assert record["height"] == 480
        assert "faces" in record
        assert "motion" in record

    # Verify summary JSON
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_frames_analyzed"] == 30
    assert summary["frames_with_face"] >= 20
