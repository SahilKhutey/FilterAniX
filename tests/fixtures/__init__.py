"""Test Fixtures Package."""
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent

CREATOR_VIDEO = FIXTURE_DIR / "creator_test_video.mp4"


def ensure_creator_video() -> Path:
    """Ensures that the synthetic creator test video fixture exists on disk."""
    if CREATOR_VIDEO.exists() and CREATOR_VIDEO.stat().st_size > 1000:
        return CREATOR_VIDEO

    from tests.fixtures.generate_creator_video import generate

    generate(CREATOR_VIDEO)

    return CREATOR_VIDEO
