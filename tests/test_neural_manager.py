"""Unit tests for NeuralAssistManager multi-rate scheduling and field injection."""
from __future__ import annotations

import numpy as np
import pytest

from src.neural import NeuralAssistConfig, NeuralAssistManager
from src.vision.models import FrameVisionData, FaceData, BoundingBox


def test_manager_disabled_behavior():
    config = NeuralAssistConfig(enabled=False)
    manager = NeuralAssistManager(config)

    vision = FrameVisionData(frame_index=0, timestamp=0.0, width=320, height=240)
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)

    out_vision = manager.process_frame(frame, vision)

    assert out_vision is vision
    assert getattr(out_vision, "neural_matte", None) is None
    assert getattr(out_vision, "neural_depth", None) is None


def test_manager_enabled_injects_fields():
    config = NeuralAssistConfig(
        enabled=True,
        use_matting=True,
        use_depth=True,
        use_flow=True,
    )
    manager = NeuralAssistManager(config)

    face = FaceData(
        face_id=0,
        landmarks=[],
        bbox=BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5),
        landmark_count=0,
    )
    vision0 = FrameVisionData(frame_index=0, timestamp=0.0, width=320, height=240, faces=[face])
    frame0 = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)

    # Process frame 0
    out0 = manager.process_frame(frame0, vision0)
    assert out0.neural_matte is not None
    assert out0.neural_matte.shape == (240, 320, 1)
    assert out0.neural_depth is not None
    assert out0.neural_depth.shape == (240, 320, 1)

    # Process frame 1 (optical flow should now be active)
    vision1 = FrameVisionData(frame_index=1, timestamp=0.033, width=320, height=240, faces=[face])
    frame1 = np.roll(frame0, shift=2, axis=1)
    out1 = manager.process_frame(frame1, vision1)

    assert out1.neural_flow is not None
    assert out1.neural_flow.shape == (240, 320, 2)
    assert "neural_enabled" in out1.neural_telemetry


def test_manager_telemetry_tracking():
    config = NeuralAssistConfig(enabled=True, use_matting=True, use_depth=True)
    manager = NeuralAssistManager(config)

    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    vision = FrameVisionData(frame_index=0, timestamp=0.0, width=160, height=120)

    manager.process_frame(frame, vision)
    snapshot = manager.memory_tracker.snapshot(enabled=True)

    assert snapshot.enabled is True
    assert snapshot.budget_limit_mb == 1024.0
    assert len(snapshot.active_models) > 0
