"""Data models for Phase 1 Foundation & Video Pipeline."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoMetadata:
    """Metadata describing an imported video file."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    has_audio: bool
    video_codec: str = "unknown"
    audio_codec: str = "none"

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def filename(self) -> str:
        return Path(self.path).name

    def summary_dict(self) -> dict:
        return {
            "Filename": self.filename,
            "Resolution": self.resolution_str,
            "FPS": f"{self.fps:.2f}",
            "Frames": str(self.frame_count),
            "Duration": f"{self.duration_sec:.2f}s",
            "Audio": "Yes" if self.has_audio else "No",
            "Video Codec": self.video_codec,
            "Audio Codec": self.audio_codec,
        }


@dataclass
class ProcessingProgress:
    """Real-time progress telemetry from the frame processing worker."""
    current_frame: int
    total_frames: int
    percent: float
    fps: float
    elapsed_sec: float
    eta_sec: float
    status_message: str = "Processing..."


@dataclass
class CameraDeviceInfo:
    """Details for an available camera input."""
    index: int
    name: str = "Camera Device"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
