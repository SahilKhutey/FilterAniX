"""Tests for UI controller callbacks and style parameter handling."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.core.jobs import JobManager
from src.ui.callbacks import (
    load_style_choices,
    inspect_input_video_file,
    pause_active_job,
    resume_active_job,
    stop_active_job,
    export_preset_video,
    save_style_preset,
    render_style_lab_frame,
    list_recent_projects,
)
from src.ui.components.parameter_panel import build_math_params_dict
from src.art.mathematical import MathematicalAnimeStyle


def test_load_style_choices():
    choices = load_style_choices()
    assert isinstance(choices, list)
    assert len(choices) > 0
    assert "anime_creator" in choices


def test_inspect_video_empty():
    res, fps, frames, dur, audio = inspect_input_video_file(None)
    assert res == "-"
    assert fps == "-"
    assert audio == "No"


def test_build_math_params_dict():
    params = build_math_params_dict(
        contrast=1.12,
        gamma=0.98,
        tone_strength=0.85,
        saturation=1.15,
        palette_mix=0.55,
        color_levels=16,
        palette_temp=0.68,
        smooth_sigma=1.2,
        texture_suppress=0.70,
        detail_retention=0.30,
        edge_strength=0.75,
        edge_threshold=0.15,
        edge_softness=0.06,
        line_darkness=0.85,
        shadow_thresh=0.42,
        shadow_strength=0.22,
        highlight_thresh=0.76,
        highlight_strength=0.12,
        warm_light=0.20,
        face_strength=0.92,
        pose_strength=0.78,
        hand_strength=0.82,
        bg_simplify=0.60,
        temporal_strength=0.14,
        motion_limit=0.16,
    )
    assert params["contrast"] == 1.12
    assert params["color_levels"] == 16
    assert params["edge_strength"] == 0.75

    # Verify that MathematicalAnimeStyle accepts these params
    style = MathematicalAnimeStyle(**params).validated()
    assert style.contrast == 1.12
    assert style.edge_strength == 0.75


def test_save_style_preset(tmp_path, monkeypatch):
    test_styles_file = tmp_path / "styles.json"
    monkeypatch.chdir(tmp_path)

    params = {"contrast": 1.10, "edge_strength": 0.80}
    result_msg = save_style_preset("Studio Warmth", params)
    assert "saved successfully" in result_msg

    choices = load_style_choices()
    assert "studio_warmth" in choices


def test_job_control_callbacks():
    manager = JobManager(max_workers=1)
    job = manager.create()

    # Pause
    msg, btn_st = pause_active_job(manager, job.job_id)
    assert job.status == "paused"
    assert btn_st["resume_enabled"] is True

    # Resume
    msg, btn_st = resume_active_job(manager, job.job_id)
    assert job.status == "running"
    assert btn_st["pause_enabled"] is True

    # Stop
    msg, btn_st = stop_active_job(manager, job.job_id)
    assert job.status == "cancelled"
    assert btn_st["start_enabled"] is True


def test_export_preset_validation():
    # Empty master path
    out, msg = export_preset_video(None, "1080p")
    assert out is None
    assert "provide a master video" in msg

    # Non-existent file
    out, msg = export_preset_video("non_existent_master.mp4", "1080p")
    assert out is None
    assert "not found" in msg


def test_list_recent_projects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "test_proj_alpha").mkdir()

    projects = list_recent_projects()
    assert len(projects) == 1
    assert projects[0][0] == "test_proj_alpha"


def test_render_style_lab_frame(tmp_path):
    # Create small synthetic test video
    test_video = tmp_path / "style_lab_test.mp4"
    h, w = 120, 160
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_video), fourcc, 24, (w, h))
    for i in range(3):
        frame = np.full((h, w, 3), 100 + i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    math_params = {
        "contrast": 1.10,
        "gamma": 0.95,
        "edge_strength": 0.70,
        "line_darkness": 0.80,
    }

    orig, art, split, status = render_style_lab_frame(str(test_video), 1, math_params)
    assert orig is not None
    assert art is not None
    assert split is not None
    assert orig.shape == (h, w, 3)
    assert art.shape == (h, w, 3)
    assert split.shape == (h, w, 3)
    assert "Frame 2 / 3" in status
