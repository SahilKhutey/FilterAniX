"""Unit tests for Live Camera stylization pipeline and mode switching."""
from __future__ import annotations

import numpy as np
import pytest

from src.ui.callbacks import live_camera_stylize_frame


def test_live_camera_none_input():
    assert live_camera_stylize_frame(None) is None


def test_live_camera_mathematical_mode():
    h, w = 240, 320
    dummy = np.full((h, w, 3), 130, dtype=np.uint8)
    # Add dummy face-like circle
    dummy[60:180, 100:220] = [215, 180, 150]

    out = live_camera_stylize_frame(dummy, mode="🎨 MATHEMATICAL ENGINE")

    assert out is not None
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    # Not completely identical to raw input
    assert not np.array_equal(out, dummy)
    # Mean should be in healthy anime range (no crushed blacks)
    assert 40.0 < np.mean(out) < 220.0


def test_live_camera_fast_preview_mode():
    h, w = 240, 320
    dummy = np.full((h, w, 3), 130, dtype=np.uint8)
    dummy[60:180, 100:220] = [215, 180, 150]

    out = live_camera_stylize_frame(dummy, mode="⚡ FAST PREVIEW")

    assert out is not None
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert 40.0 < np.mean(out) < 220.0


def test_live_camera_dimension_preservation():
    # Large frame should resize down and back up to original
    h, w = 720, 1280
    dummy = np.full((h, w, 3), 140, dtype=np.uint8)

    out = live_camera_stylize_frame(dummy, mode="🎨 MATHEMATICAL ENGINE")

    assert out is not None
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
