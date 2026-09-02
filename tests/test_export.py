"""Automated Tests for Phase 6 YouTube Multi-Resolution Export."""
from pathlib import Path
import pytest

from tests.test_video_io import create_synthetic_test_video_with_audio
from src.core.export import YouTubeExporter, YOUTUBE_PRESETS
from src.media.ffmpeg import inspect_media


def test_youtube_exporter_720p(tmp_path):
    """Verifies that YouTubeExporter encodes valid 720p output with faststart container flags."""
    master_in = tmp_path / "master_sample.mp4"
    export_out = tmp_path / "youtube_720p_out.mp4"

    create_synthetic_test_video_with_audio(master_in, num_frames=30, width=640, height=360, fps=30)

    exporter = YouTubeExporter(preset_name="720p")
    result_path = exporter.export(input_master_path=master_in, output_export_path=export_out)

    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 1000

    details = inspect_media(result_path)
    assert details.has_video is True
    assert details.has_audio is True
    assert details.width == 1280
    assert details.height == 720
