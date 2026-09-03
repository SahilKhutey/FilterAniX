from __future__ import annotations

import json
from pathlib import Path
import tempfile
import cv2
import numpy as np
import pytest

from src.core.frame import FramePacket
from src.core.errors import (
    FilterAniXError,
    InputError,
    VisionError,
    RenderingError,
    TemporalError,
    AudioError,
    CompositionError,
    ExportError,
    ValidationError,
    write_error_manifest,
)
from src.core.logging import get_stage_logger
from src.art.mathematical import (
    GeometryBox,
    GeometryObservation,
    MathematicalAnimeStyle,
    MathematicalRenderer,
    MathematicalRenderResult,
)


def test_frame_packet_valid():
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_index=0,
        timestamp_seconds=0.0,
        fps=24.0,
        rgb=img,
        width=128,
        height=64,
    )
    assert packet.width == 128
    assert packet.height == 64
    assert packet.metadata == {}


def test_frame_packet_dimension_mismatch():
    img = np.zeros((64, 128, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        FramePacket(
            frame_index=0,
            timestamp_seconds=0.0,
            fps=24.0,
            rgb=img,
            width=64,  # mismatch
            height=64,
        )


def test_frame_packet_invalid_channels():
    img = np.zeros((64, 128, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        FramePacket(
            frame_index=0,
            timestamp_seconds=0.0,
            fps=24.0,
            rgb=img,
            width=128,
            height=64,
        )


def test_error_manifest_writer(tmp_path):
    manifest_file = tmp_path / "error.json"
    err = RenderingError("Test rendering failure")
    res = write_error_manifest(
        output_path=manifest_file,
        stage="phase3_render",
        error=err,
        frame_index=42,
        input_path="sample.mp4",
        details={"code": 500},
    )
    assert res.exists()
    with open(res, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["status"] == "failed"
    assert data["stage"] == "phase3_render"
    assert data["frame_index"] == 42
    assert data["error_type"] == "RenderingError"
    assert "Test rendering failure" in data["message"]


def test_stage_logger(tmp_path):
    logger = get_stage_logger("test_stage", log_dir=tmp_path)
    logger.info("Test message for stage logger")
    log_file = tmp_path / "test_stage.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test message for stage logger" in content


def test_mathematical_renderer_single_frame():
    renderer = MathematicalRenderer(MathematicalAnimeStyle.creator_anime())
    img = np.full((128, 128, 3), 120, dtype=np.uint8)
    geom = GeometryObservation(
        width=128,
        height=128,
        face_box=GeometryBox(x0=30, y0=20, x1=90, y1=80, confidence=1.0),
    )

    result = renderer.render(img, vision=geom)
    assert isinstance(result, MathematicalRenderResult)
    assert result.output_rgb.shape == (128, 128, 3)
    assert result.output_rgb.dtype == np.uint8
    assert result.mth02 is not None
    assert result.mth10 is not None


def test_mathematical_renderer_video(tmp_path):
    # Create a small 6-frame synthetic video
    video_path = tmp_path / "input.mp4"
    out_path = tmp_path / "output.mp4"
    report_path = tmp_path / "report.json"

    fps = 24.0
    width, height = 64, 64
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for i in range(6):
        frame = np.full((height, width, 3), 80 + i * 15, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    renderer = MathematicalRenderer(MathematicalAnimeStyle.creator_anime())
    res_path = renderer.render_video(
        input_path=video_path,
        output_path=out_path,
        quality_report_path=report_path,
    )

    assert res_path.exists()
    assert report_path.exists()

    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["frames_processed"] == 6
    assert rep["status"] == "success"
