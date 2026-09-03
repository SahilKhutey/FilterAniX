"""Unit tests for MTH-02..MTH-10 intermediate stage observability and diagnostics."""
from __future__ import annotations

import numpy as np
import pytest

from src.art.mathematical import MathematicalAnimeEngine, MathematicalAnimeStyle


def test_render_stages_returns_all_mth_fields():
    engine = MathematicalAnimeEngine()
    input_rgb = np.random.randint(60, 200, (120, 160, 3), dtype=np.uint8)

    out_rgb, stages, telemetry = engine.render_stages(input_rgb, stabilize=False)

    assert out_rgb.shape == (120, 160, 3)
    assert out_rgb.dtype == np.uint8

    expected_stages = [
        "Input",
        "MTH-02 Color",
        "MTH-03 Tone",
        "MTH-04 Palette",
        "MTH-05 Edge",
        "MTH-06 Shadow/Highlight",
        "MTH-07 Geometry",
        "MTH-08 Face",
        "MTH-09 Lighting",
        "MTH-10 Temporal",
        "Final",
    ]

    for stage_name in expected_stages:
        assert stage_name in stages, f"Missing stage {stage_name}"
        stage_img = stages[stage_name]
        assert stage_img.shape == (120, 160, 3)
        assert stage_img.dtype == np.uint8


def test_diagnostics_report_formatting():
    engine = MathematicalAnimeEngine()
    input_rgb = np.full((120, 160, 3), 120, dtype=np.uint8)
    out_rgb, stages, telemetry = engine.render_stages(input_rgb, stabilize=False)

    report = engine.format_diagnostics_report(telemetry, frame_index=42)

    assert "Frame: 42" in report
    assert "MTH-02 Color Field" in report
    assert "MTH-03 Tone Field" in report
    assert "MTH-04 Palette Field" in report
    assert "MTH-05 Edge Field" in report
    assert "MTH-06 Shadow/Highlight" in report
    assert "MTH-07 Geometry" in report
    assert "MTH-08 Face" in report
    assert "MTH-09 Lighting" in report
    assert "MTH-10 Temporal" in report
    assert "FINAL" in report
