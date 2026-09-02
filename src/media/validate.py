"""Media Output Validator and A/V Synchronization Auditor."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.media.ffmpeg import inspect_media, MediaDetails


@dataclass
class ValidationResult:
    """Report detailing output video health and audio/video sync integrity."""
    valid: bool
    path: str
    video_duration: float
    audio_duration: float
    drift_seconds: float
    has_video: bool
    has_audio: bool
    resolution: str
    fps: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "path": self.path,
            "resolution": self.resolution,
            "fps": round(self.fps, 2),
            "video_duration": round(self.video_duration, 3),
            "audio_duration": round(self.audio_duration, 3),
            "drift_seconds": round(self.drift_seconds, 3),
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "warnings": self.warnings,
        }


class OutputValidator:
    """Validates that final MP4 container is healthy, playable, and audio/video synchronized."""

    def __init__(self, max_drift_seconds: float = 0.50):
        self.max_drift_seconds = max_drift_seconds

    def validate(self, media_path: str | Path) -> ValidationResult:
        media_p = Path(media_path).resolve()
        if not media_p.exists() or media_p.stat().st_size < 1000:
            return ValidationResult(
                valid=False,
                path=str(media_p),
                video_duration=0.0,
                audio_duration=0.0,
                drift_seconds=0.0,
                has_video=False,
                has_audio=False,
                resolution="0x0",
                fps=0.0,
                warnings=["File does not exist or is too small/corrupted."],
            )

        details: MediaDetails = inspect_media(media_p)
        warnings = []
        is_valid = True

        if not details.has_video:
            warnings.append("No video stream found in container.")
            is_valid = False

        drift = 0.0
        if details.has_audio and details.has_video:
            drift = abs(details.video_duration - details.audio_duration)
            if drift > self.max_drift_seconds:
                warnings.append(
                    f"A/V duration drift ({drift:.2f}s) exceeds threshold ({self.max_drift_seconds:.2f}s)."
                )

        return ValidationResult(
            valid=is_valid,
            path=str(media_p),
            video_duration=details.video_duration,
            audio_duration=details.audio_duration,
            drift_seconds=drift,
            has_video=details.has_video,
            has_audio=details.has_audio,
            resolution=details.resolution_str,
            fps=details.fps,
            warnings=warnings,
        )
