"""Automated Tests for Phase 3 Artistic Style Engine."""
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.art.types import StylePreset, RenderConfig, RendererBackend
from src.art.preprocess import ControlBuilder
from src.art.temporal import TemporalStabilizer
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.art.style_engine import StyleEngine
from src.art.video_renderer import VideoStyleRenderer
from src.vision.models import FrameVisionData, FaceData, Landmark, BoundingBox


def test_control_builder():
    """Verifies that the ControlBuilder constructs edge and combined control maps."""
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 150), (255, 255, 255), -1)

    builder = ControlBuilder()
    control_map = builder.build_control_map(img)

    assert control_map.edge_map.shape == (200, 300, 3)
    assert control_map.combined_control.shape == (200, 300, 3)
    assert np.any(control_map.edge_map > 0)


def test_temporal_stabilizer_scene_cut():
    """Verifies that the TemporalStabilizer detects scene cuts and resets history."""
    stabilizer = TemporalStabilizer(scene_cut_threshold=0.30)

    # Frame 1: Bright image
    f1_raw = np.full((100, 100, 3), 240, dtype=np.uint8)
    f1_art = np.full((100, 100, 3), 220, dtype=np.uint8)
    res1 = stabilizer.stabilize(f1_raw, f1_art)
    assert np.all(res1 == f1_art)

    # Frame 2: Smooth continuation
    f2_raw = np.full((100, 100, 3), 235, dtype=np.uint8)
    f2_art = np.full((100, 100, 3), 215, dtype=np.uint8)
    res2 = stabilizer.stabilize(f2_raw, f2_art)
    assert stabilizer.is_scene_cut(f2_raw) is False

    # Frame 3: Hard scene cut (Complete blackout)
    f3_raw = np.full((100, 100, 3), 10, dtype=np.uint8)
    f3_art = np.full((100, 100, 3), 15, dtype=np.uint8)
    assert stabilizer.is_scene_cut(f3_raw) is True
    res3 = stabilizer.stabilize(f3_raw, f3_art)
    assert np.all(res3 == f3_art)  # Did not blend with previous bright frame!


def test_style_engine_end_to_end():
    """Verifies that the StyleEngine transforms a raw image cleanly."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = [190, 200, 210]
    cv2.circle(img, (160, 120), 40, (230, 190, 170), -1)

    engine = StyleEngine()
    art = engine.render_frame(img, stabilize=False)

    assert art.shape == (240, 320, 3)
    assert art.dtype == np.uint8
