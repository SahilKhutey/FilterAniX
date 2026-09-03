"""Unit tests for content-aware semantic hierarchy in MathematicalAnimeCompositor."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.art.mathematical import (
    MathematicalAnimeCompositor,
    MathematicalAnimeEngine,
    MathematicalAnimeStyle,
    compute_semantic_face_masks,
)
from src.vision.models import FrameVisionData, FaceData, BoundingBox


def test_semantic_sub_masks_separation():
    h, w = 240, 320
    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5, confidence=0.95),
        landmark_count=0,
        mouth_opening=0.0,
    )

    face_m, hair_m, eye_m, mouth_m, skin_m = compute_semantic_face_masks(h, w, face)

    assert face_m.shape == (h, w, 1)
    assert hair_m.shape == (h, w, 1)
    assert eye_m.shape == (h, w, 1)
    assert mouth_m.shape == (h, w, 1)
    assert skin_m.shape == (h, w, 1)

    # Hair should have weight above face center
    assert np.max(hair_m) > 0.5
    # Eye mask should have localized weight
    assert np.max(eye_m) > 0.5
    # Skin mask should have positive weight on face
    assert np.max(skin_m) > 0.5


def test_compositor_face_preservation_no_black_blob():
    """Verifies that eye/mouth regions never collapse to black blobs."""
    h, w = 240, 320
    img = np.full((h, w, 3), 160, dtype=np.uint8)
    # Draw face
    cv2.ellipse(img, (160, 120), (50, 70), 0, 0, 360, (230, 190, 165), -1)
    # Eyes
    cv2.circle(img, (140, 110), 6, (60, 45, 40), -1)
    cv2.circle(img, (180, 110), 6, (60, 45, 40), -1)
    cv2.circle(img, (142, 108), 2, (255, 255, 255), -1)
    cv2.circle(img, (182, 108), 2, (255, 255, 255), -1)

    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.25, y=0.15, width=0.50, height=0.60, confidence=0.99),
        landmark_count=0,
        mouth_opening=0.0,
    )
    vision = FrameVisionData(frame_index=1, timestamp=0.0, width=w, height=h, faces=[face])

    engine = MathematicalAnimeEngine()
    out, stages, telemetry = engine.render_stages(img, vision_data=vision, stabilize=False)

    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8

    # Check eye region in output: must not be solid black 0
    eye_crop = out[104:116, 134:146]
    assert np.mean(eye_crop) > 30.0, "Eye collapsed to black blob!"

    # Preservation metrics must be present and report high fidelity
    audit = telemetry.get("quality_audit")
    assert audit is not None
    assert audit.p_structure > 0.70
    assert audit.p_face > 0.75
    assert audit.q_score > 0.75
