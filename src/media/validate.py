from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.media.ffmpeg import require_ffmpeg, inspect_media, MediaDetails


def probe(path: str | Path) -> dict[str, Any]:
    """Inspects container streams and format via FFprobe JSON output with FFmpeg fallback."""
    path_str = str(Path(path).resolve())
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Media file not found: {path_str}")

    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        try:
            command = [
                ffprobe_bin,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                path_str,
            ]
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            return json.loads(result.stdout)
        except Exception:
            pass

    # Reliable fallback using FFmpeg header output and OpenCV metadata
    details = inspect_media(path_str)
    streams = []
    if details.has_video:
        streams.append({
            "codec_type": "video",
            "codec_name": details.video_codec,
            "width": details.width,
            "height": details.height,
            "r_frame_rate": f"{int(details.fps)}/1",
            "duration": str(details.video_duration),
        })
    if details.has_audio:
        streams.append({
            "codec_type": "audio",
            "codec_name": details.audio_codec,
            "sample_rate": str(details.sample_rate),
            "channels": details.channels,
            "duration": str(details.audio_duration),
        })

    max_dur = max(details.video_duration, details.audio_duration)
    return {
        "streams": streams,
        "format": {
            "filename": path_str,
            "duration": str(max_dur),
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
        },
    }


def validate_video(path: str | Path) -> dict[str, Any]:
    """Validates video stream health, dimensions, and audio/video duration alignment."""
    data = probe(path)

    streams = data.get("streams", [])

    video = [
        s for s in streams
        if s.get("codec_type") == "video"
    ]

    audio = [
        s for s in streams
        if s.get("codec_type") == "audio"
    ]

    errors = []
    warnings = []

    if not video:
        errors.append("No video stream found.")

    width = 0
    height = 0
    if video:
        width = int(video[0].get("width", 0))
        height = int(video[0].get("height", 0))

        if width <= 0 or height <= 0:
            errors.append("Invalid video dimensions.")

    video_duration = float(video[0].get("duration", 0)) if video and video[0].get("duration") else 0.0
    audio_duration = float(audio[0].get("duration", 0)) if audio and audio[0].get("duration") else 0.0

    if video_duration <= 0 and data.get("format", {}).get("duration"):
        video_duration = float(data["format"]["duration"])
    if audio_duration <= 0 and audio and data.get("format", {}).get("duration"):
        audio_duration = float(data["format"]["duration"])

    duration_delta = abs(video_duration - audio_duration) if (video and audio) else 0.0

    if video and audio and duration_delta > 0.35:
        warnings.append(
            f"A/V duration difference: {duration_delta:.3f}s"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "width": width,
        "height": height,
        "video_duration": round(video_duration, 3),
        "audio_duration": round(audio_duration, 3),
        "duration_delta": round(duration_delta, 3),
    }


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
