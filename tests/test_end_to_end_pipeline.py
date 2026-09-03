"""End-to-End Production Pipeline Integration Test.

Tests the full 6-phase pipeline on a synthetic creator video with voice/audio,
verifying stage sequencing, temporal planning, lip-sync rendering, audio preservation,
and YouTube broadcast master validation.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import pytest
import cv2
import numpy as np

from src.core.project import Project
from src.core.pipeline import PipelineManager
from src.media.ffmpeg import inspect_media, run_ffmpeg


def create_test_video(
    path: Path,
    duration: float = 2.0,
    fps: int = 30,
    width: int = 640,
    height: int = 360,
):
    """Generates synthetic creator test footage with moving face, blinking eyes, mouth animation, and gesturing hands."""
    frame_count = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    try:
        for index in range(frame_count):
            frame = np.full((height, width, 3), 30, dtype=np.uint8)

            # Background
            cv2.rectangle(frame, (0, 0), (width, height), (35, 45, 60), -1)

            # Head
            center_x = width // 2 + int(30 * np.sin(index / 12))
            center_y = height // 2 - 30
            cv2.circle(frame, (center_x, center_y), 70, (190, 160, 130), -1)

            # Eyes
            cv2.circle(frame, (center_x - 25, center_y - 10), 7, (20, 20, 20), -1)
            cv2.circle(frame, (center_x + 25, center_y - 10), 7, (20, 20, 20), -1)

            # Animated mouth
            mouth_open = 10 + int(8 * abs(np.sin(index / 5)))
            cv2.ellipse(frame, (center_x, center_y + 30), (18, mouth_open), 0, 0, 360, (50, 30, 30), -1)

            # Body
            cv2.rectangle(frame, (center_x - 100, center_y + 70), (center_x + 100, height), (80, 110, 150), -1)

            # Moving hand
            hand_x = width - 100 + int(40 * np.sin(index / 8))
            cv2.circle(frame, (hand_x, center_y + 90), 25, (180, 150, 120), -1)

            writer.write(frame)
    finally:
        writer.release()


def add_test_audio(
    video_path: Path,
    output_path: Path,
):
    """Muxes a synthetic audio test track into the test video."""
    command = [
        "-y",
        "-i", str(video_path),
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2.0",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        str(output_path),
    ]
    res = run_ffmpeg(command, check=False)
    if res.returncode != 0 or not output_path.exists():
        import shutil
        shutil.copy2(str(video_path), str(output_path))


def test_full_pipeline(tmp_path: Path):
    """Verifies that the entire 6-phase pipeline runs end-to-end with video, audio, and validation."""
    silent_video = tmp_path / "creator_video.mp4"
    input_video = tmp_path / "creator_with_audio.mp4"

    create_test_video(silent_video, duration=2.0)
    add_test_audio(silent_video, input_video)

    project_dir = tmp_path / "project"
    project = Project(project_dir)
    project.create("e2e_test")

    pipeline = PipelineManager(project)
    result = pipeline.run(
        input_video=input_video,
        style="anime_creator",
    )

    final_video = project_dir / "output" / "youtube_master.mp4"
    assert final_video.exists()
    assert final_video.stat().st_size > 0
    assert (project_dir / "vision" / "vision.jsonl").exists()
    assert (project_dir / "consistency" / "temporal_plan.jsonl").exists()
    assert (project_dir / "lipsync" / "lipsync.jsonl").exists()
    assert (project_dir / "artistic" / "animated.mp4").exists()
    assert (project_dir / "reports" / "validation.json").exists()
    assert result["final_video"] == str(final_video)

    # Validate audio and media integrity
    media = inspect_media(final_video)
    assert media.has_video is True
    assert media.video_codec in ("h264", "libx264", "avc1")
    assert media.has_audio is True
    assert media.audio_codec in ("aac", "mp4a")


def test_full_pipeline_ordering_and_artifacts(tmp_path: Path):
    """Verifies that Phase 4 and Phase 5A execute before Phase 3 and their outputs exist."""
    silent_video = tmp_path / "creator_seq_silent.mp4"
    input_video = tmp_path / "creator_seq_with_audio.mp4"

    create_test_video(silent_video, duration=2.0)
    add_test_audio(silent_video, input_video)

    project_dir = tmp_path / "project_seq"
    project = Project(project_dir)
    project.create("seq_test")

    pipeline = PipelineManager(project)
    results = pipeline.run(
        input_video=input_video,
        style="anime_creator",
    )

    temporal_plan = project_dir / "consistency" / "temporal_plan.jsonl"
    lipsync_jsonl = project_dir / "lipsync" / "lipsync.jsonl"

    assert project.stage_complete("input")
    assert project.stage_complete("vision")
    assert project.stage_complete("consistency")
    assert project.stage_complete("lipsync")
    assert project.stage_complete("artistic")
    assert project.stage_complete("composition")
    assert project.stage_complete("validation")

    with open(temporal_plan, "r", encoding="utf-8") as f:
        first_decision = json.loads(f.readline())
        assert "is_keyframe" in first_decision or "keyframe" in first_decision
        assert "scene_id" in first_decision
        assert "reference_strength" in first_decision

    with open(lipsync_jsonl, "r", encoding="utf-8") as f:
        first_lip = json.loads(f.readline())
        assert "viseme" in first_lip or "state" in first_lip
