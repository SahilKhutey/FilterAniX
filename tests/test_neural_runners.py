"""Unit tests for individual assistive neural runners with fallback behavior."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.neural.segmentation.u2netp import U2NetpRunner
from src.neural.segmentation.modnet import MODNetRunner
from src.neural.motion.micron_flow import MicronFlowRunner
from src.neural.depth.depth_anything import DepthAnythingRunner
from src.vision.models import BoundingBox


def test_u2netp_runner_output_contract():
    runner = U2NetpRunner()
    h, w = 180, 240
    frame = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)
    bbox = BoundingBox(x=0.25, y=0.15, width=0.5, height=0.6)

    mask, latency = runner.predict_mask(frame, fallback_hint_bbox=bbox)

    assert mask.shape == (h, w, 1)
    assert mask.dtype == np.float32
    assert float(np.min(mask)) >= 0.0
    assert float(np.max(mask)) <= 1.0
    assert latency >= 0.0


def test_modnet_runner_output_contract():
    runner = MODNetRunner()
    h, w = 180, 240
    frame = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)
    bbox = BoundingBox(x=0.3, y=0.2, width=0.4, height=0.5)

    matte, latency = runner.predict_matte(frame, fallback_face_bbox=bbox)

    assert matte.shape == (h, w, 1)
    assert matte.dtype == np.float32
    assert float(np.min(matte)) >= 0.0
    assert float(np.max(matte)) <= 1.0
    assert latency >= 0.0


def test_micron_flow_runner_output_contract():
    runner = MicronFlowRunner()
    h, w = 180, 240
    prev = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)
    # Simulate slight rightward motion
    curr = np.roll(prev, shift=2, axis=1)

    flow, latency = runner.estimate_flow(prev, curr)

    assert flow.shape == (h, w, 2)
    assert flow.dtype == np.float32
    assert latency >= 0.0


def test_depth_anything_runner_output_contract():
    runner = DepthAnythingRunner()
    h, w = 180, 240
    frame = np.random.randint(40, 220, (h, w, 3), dtype=np.uint8)

    depth, latency = runner.estimate_depth(frame)

    assert depth.shape == (h, w, 1)
    assert depth.dtype == np.float32
    assert float(np.min(depth)) >= 0.0
    assert float(np.max(depth)) <= 1.0
    assert latency >= 0.0
