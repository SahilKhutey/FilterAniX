"""Phase 5 Media Package."""
from src.media.ffmpeg import (
    require_ffmpeg,
    require_ffprobe,
    run_ffmpeg,
    run_ffprobe,
    MediaDetails,
    inspect_media,
    normalize_audio,
    mux_media,
)
from src.media.compose import compose_final_video, VideoCompositor
from src.media.validate import probe, validate_video, ValidationResult, OutputValidator

__all__ = [
    "require_ffmpeg",
    "require_ffprobe",
    "run_ffmpeg",
    "run_ffprobe",
    "MediaDetails",
    "inspect_media",
    "normalize_audio",
    "mux_media",
    "compose_final_video",
    "VideoCompositor",
    "probe",
    "validate_video",
    "ValidationResult",
    "OutputValidator",
]
