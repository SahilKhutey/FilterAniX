"""Automated Tests for Phase 5 Media Composition and Output Validation."""
from pathlib import Path
import pytest

from tests.test_video_io import create_synthetic_test_video_with_audio
from src.media.ffmpeg import inspect_media
from src.media.compose import VideoCompositor
from src.media.validate import OutputValidator


def test_media_inspection_and_validation(tmp_path):
    """Verifies that media inspection and output validation accurately report streams."""
    source_video = tmp_path / "source_test.mp4"
    create_synthetic_test_video_with_audio(source_video, num_frames=30, width=320, height=240, fps=30)

    details = inspect_media(source_video)
    assert details.has_video is True
    assert details.has_audio is True
    assert details.width == 320
    assert details.height == 240

    validator = OutputValidator()
    report = validator.validate(source_video)
    assert report.valid is True
    assert report.has_video is True
    assert report.has_audio is True
    assert report.drift_seconds < 0.20
