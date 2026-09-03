"""Unit tests for FilterAniX Objective Preservation and Quality Metrics."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.art.mathematical.preservation_metrics import (
    PreservationMetricsEngine,
    FrameQualityAudit,
)
from src.vision.models import FrameVisionData, FaceData, BoundingBox


def test_structural_preservation_identical_frame():
    engine = PreservationMetricsEngine()
    frame = np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8)

    p_struct = engine.compute_structural_preservation(frame, frame)
    assert p_struct > 0.95


def test_face_preservation_healthy_render():
    engine = PreservationMetricsEngine()
    h, w = 120, 160
    src = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.circle(src, (80, 60), 30, (220, 180, 160), -1)

    trans = src.copy()
    # Apply warm tone
    trans[:, :, 0] = np.clip(trans[:, :, 0] + 15, 0, 255)

    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5, confidence=0.99),
        landmark_count=0,
        mouth_opening=0.0,
    )
    vision = FrameVisionData(frame_index=0, timestamp=0.0, width=w, height=h, faces=[face])

    p_face = engine.compute_face_preservation(src, trans, vision)
    assert p_face > 0.80


def test_face_preservation_penalizes_crushed_face():
    engine = PreservationMetricsEngine()
    h, w = 120, 160
    src = np.full((h, w, 3), 150, dtype=np.uint8)
    cv2.circle(src, (80, 60), 30, (220, 180, 160), -1)

    # Crushed face to near-black
    trans_crushed = src.copy()
    trans_crushed[24:84, 48:112] = 10

    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5, confidence=0.99),
        landmark_count=0,
        mouth_opening=0.0,
    )
    vision = FrameVisionData(frame_index=0, timestamp=0.0, width=w, height=h, faces=[face])

    p_face = engine.compute_face_preservation(src, trans_crushed, vision)
    assert p_face <= 0.30


def test_temporal_stability_motion_tolerance():
    engine = PreservationMetricsEngine()
    frame_a = np.full((120, 160, 3), 140, dtype=np.uint8)
    frame_b = np.full((120, 160, 3), 145, dtype=np.uint8)

    # When motion score accounts for the slight difference, stability remains high
    s_temp = engine.compute_temporal_stability(frame_a, frame_b, motion_score=0.05)
    assert s_temp > 0.85


def test_evaluate_frame_generates_composite_q():
    engine = PreservationMetricsEngine()
    src = np.random.randint(60, 180, (120, 160, 3), dtype=np.uint8)
    trans = np.clip(src * 1.1 + 5, 0, 255).astype(np.uint8)

    audit = engine.evaluate_frame(src, trans)
    assert isinstance(audit, FrameQualityAudit)
    assert 0.0 <= audit.q_score <= 1.0
    assert 0.0 <= audit.p_structure <= 1.0
    assert 0.0 <= audit.d_color <= 1.0
    audit_dict = audit.to_dict()
    assert "q_score" in audit_dict
