"""Unit tests verifying mathematical compositor fusion with neural assist fields."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.art.mathematical import (
    MathematicalAnimeCompositor,
    MathematicalAnimeEngine,
    MathematicalAnimeStyle,
    validate_render,
)
from src.vision.models import FrameVisionData, FaceData, BoundingBox


def test_compositor_consumes_neural_matte_and_depth():
    h, w = 180, 240
    frame = np.full((h, w, 3), 140, dtype=np.uint8)
    cv2.circle(frame, (120, 90), 40, (230, 190, 160), -1)

    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5),
        landmark_count=0,
    )
    vision = FrameVisionData(frame_index=0, timestamp=0.0, width=w, height=h, faces=[face])

    # Inject simulated neural fields
    fake_matte = np.zeros((h, w, 1), dtype=np.float32)
    fake_matte[50:130, 80:160] = 1.0
    vision.neural_matte = fake_matte

    fake_depth = np.linspace(0.2, 0.9, h, dtype=np.float32)[:, np.newaxis, np.newaxis]
    fake_depth = np.tile(fake_depth, (1, w, 1))
    vision.neural_depth = fake_depth

    engine = MathematicalAnimeEngine()
    rendered = engine.render(frame, vision_data=vision, stabilize=False)

    assert rendered.shape == (h, w, 3)
    assert rendered.dtype == np.uint8
    # Quality gate must pass
    assert validate_render(rendered, frame) is True


def test_compositor_flow_guided_stabilization():
    h, w = 180, 240
    frame0 = np.full((h, w, 3), 120, dtype=np.uint8)
    cv2.rectangle(frame0, (80, 60), (140, 120), (200, 100, 50), -1)

    # Frame 1 shifted by 2 pixels right
    frame1 = np.roll(frame0, shift=2, axis=1)

    vision1 = FrameVisionData(frame_index=1, timestamp=0.033, width=w, height=h)
    # Provide synthetic flow field (dx = 2.0, dy = 0.0)
    flow = np.zeros((h, w, 2), dtype=np.float32)
    flow[:, :, 0] = 2.0
    vision1.neural_flow = flow

    engine = MathematicalAnimeEngine()
    _ = engine.render(frame0, vision_data=None, stabilize=True)
    out1 = engine.render(frame1, vision_data=vision1, stabilize=True)

    assert out1.shape == (h, w, 3)
    assert out1.dtype == np.uint8
    assert validate_render(out1, frame1) is True
