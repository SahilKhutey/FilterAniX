"""Tests for Synthetic Creator Video Fixture."""
from pathlib import Path
import cv2

from tests.fixtures import ensure_creator_video


def test_creator_fixture_exists():
    """Verifies that the synthetic creator test video fixture can be retrieved or generated."""
    video = ensure_creator_video()
    assert video.exists(), f"Missing synthetic creator fixture: {video}"


def test_creator_fixture_is_readable():
    """Verifies that the fixture is a valid readable MP4 with expected dimensions and frame counts."""
    video = ensure_creator_video()
    capture = cv2.VideoCapture(str(video))

    try:
        assert capture.isOpened(), f"Could not open fixture: {video}"

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        assert frame_count >= 150
        assert 24 <= fps <= 60
        assert width >= 640
        assert height >= 480

    finally:
        capture.release()
