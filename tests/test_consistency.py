"""Automated Tests for Phase 4 Consistency, Temporal Planning, and Identity Auditing."""
import json
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.consistency.types import ReferenceProfile, RenderDecision
from src.consistency.identity import IdentityProfileBuilder, IdentityScorer
from src.consistency.scene import SceneDetector
from src.consistency.motion import MotionAnalyzer
from src.consistency.controller import TemporalController
from src.consistency.planner import TemporalPlanner
from src.consistency.report import ConsistencyAuditor
from src.vision.models import MotionData


def test_identity_profile_and_scoring():
    """Verifies that reference profiles are extracted accurately and score appropriately."""
    ref_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ref_img[:] = [200, 150, 100]
    cv2.circle(ref_img, (50, 50), 30, (50, 100, 200), -1)

    profile = IdentityProfileBuilder.build_profile(ref_img, name="test_char")
    assert profile.name == "test_char"
    assert len(profile.dominant_palette) > 0

    scorer = IdentityScorer(profile)

    # Identical image should score very high (> 0.90)
    metrics_self = scorer.evaluate_frame(ref_img)
    assert metrics_self.similarity > 0.90
    assert metrics_self.warning is False

    # Completely different image (black) should score significantly lower
    diff_img = np.zeros((100, 100, 3), dtype=np.uint8)
    metrics_diff = scorer.evaluate_frame(diff_img)
    assert metrics_diff.similarity < metrics_self.similarity


def test_scene_detector_and_controller():
    """Verifies that the controller assigns keyframes and detects scene cuts."""
    controller = TemporalController(keyframe_interval=5)

    f1 = np.full((50, 50, 3), 100, dtype=np.uint8)
    f2 = np.full((50, 50, 3), 102, dtype=np.uint8)
    f_cut = np.full((50, 50, 3), 10, dtype=np.uint8)

    # Frame 0: Initial Keyframe
    d0 = controller.evaluate_frame(0, 0.0, f1)
    assert d0.is_keyframe is True
    assert d0.reason == "initial_keyframe"

    # Frame 1: Intermediate neighbor
    d1 = controller.evaluate_frame(1, 0.033, f2)
    assert d1.is_keyframe is False
    assert d1.preserve_previous is True

    # High motion injection
    d_motion = controller.evaluate_frame(2, 0.066, f2, motion_data=MotionData(mean_magnitude=8.0, moving_pixel_ratio=0.5, valid=True))
    assert d_motion.is_keyframe is True
    assert d_motion.reason == "high_motion_keyframe"

    # Hard scene cut
    d_cut = controller.evaluate_frame(3, 0.099, f_cut)
    assert d_cut.is_scene_cut is True
    assert d_cut.is_keyframe is True
    assert d_cut.preserve_previous is False
    assert d_cut.scene_id > 0


def test_temporal_planner_and_auditor(tmp_path):
    """Verifies that TemporalPlanner and ConsistencyAuditor work end-to-end."""
    test_video = tmp_path / "test_consistency_vid.mp4"
    plan_path = tmp_path / "test_plan.jsonl"
    report_path = tmp_path / "test_report.json"

    # Create short 15-frame video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(test_video), fourcc, 30.0, (160, 120))
    for i in range(15):
        frame = np.full((120, 160, 3), 120 + i * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # 1. Build Plan
    planner = TemporalPlanner(keyframe_interval=5)
    out_plan = planner.generate_plan(video_path=test_video, output_plan_path=plan_path)
    assert Path(out_plan).exists()

    with open(out_plan, "r") as f:
        lines = f.readlines()
        assert len(lines) == 15

    # 2. Build Reference & Audit
    ref_img = np.full((120, 160, 3), 130, dtype=np.uint8)
    profile = IdentityProfileBuilder.build_profile(ref_img)

    auditor = ConsistencyAuditor(profile)
    report = auditor.audit_video(video_path=test_video, output_report_path=report_path)

    assert report.frames == 15
    assert report.mean_similarity > 0.70
    assert Path(report_path).exists()
