"""Advanced FFmpeg Media Processing, Loudness Normalization, and Subtitle Multiplexing."""
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2

from src.io.video_io import get_ffmpeg_executable, get_ffprobe_executable


@dataclass
class MediaDetails:
    """Detailed technical report of container streams."""
    path: str
    video_codec: str = "unknown"
    audio_codec: str = "none"
    width: int = 0
    height: int = 0
    fps: float = 30.0
    sample_rate: int = 48000
    channels: int = 2
    video_duration: float = 0.0
    audio_duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "resolution": self.resolution_str,
            "fps": round(self.fps, 2),
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "video_duration": round(self.video_duration, 3),
            "audio_duration": round(self.audio_duration, 3),
            "has_video": self.has_video,
            "has_audio": self.has_audio,
        }


def inspect_media(path: str | Path) -> MediaDetails:
    """Inspects video and audio stream parameters using FFmpeg."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    ffmpeg_bin = get_ffmpeg_executable()
    proc = subprocess.run(
        [ffmpeg_bin, "-i", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

    details = MediaDetails(path=str(path))

    # Video stream detection
    v_match = re.search(r"Stream #\d+:\d+.*Video:\s*([a-zA-Z0-9_\-]+).*?(\d{2,5})x(\d{2,5})", output_text)
    if v_match:
        details.has_video = True
        details.video_codec = v_match.group(1)
        details.width = int(v_match.group(2))
        details.height = int(v_match.group(3))

    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", output_text)
    if fps_match:
        details.fps = float(fps_match.group(1))

    # Audio stream detection
    a_match = re.search(r"Stream #\d+:\d+.*Audio:\s*([a-zA-Z0-9_\-]+).*?(\d+)\s*Hz.*?(\bmono\b|\bstereo\b|\d+\s*channels)", output_text, re.IGNORECASE)
    if a_match:
        details.has_audio = True
        details.audio_codec = a_match.group(1)
        details.sample_rate = int(a_match.group(2))
        chan_str = a_match.group(3).lower()
        details.channels = 1 if "mono" in chan_str else 2

    # Duration parsing
    dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output_text)
    if dur_match:
        h, m, s = map(float, dur_match.groups())
        dur_val = h * 3600 + m * 60 + s
        details.video_duration = dur_val
        if details.has_audio:
            details.audio_duration = dur_val

    # Backup OpenCV inspection for video stats if FFmpeg regex missed
    if details.width == 0 or details.height == 0:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            details.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            details.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            details.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if details.video_duration <= 0 and details.fps > 0:
                details.video_duration = fc / details.fps
            details.has_video = True
            cap.release()

    return details


def normalize_audio(
    input_audio_source: str | Path,
    output_audio_path: str | Path,
    target_lufs: float = -14.0,
    true_peak: float = -1.5,
) -> bool:
    """Normalizes audio track to YouTube / EBU R128 loudness standards (-14 LUFS, -1.5 dBTP)."""
    ffmpeg_bin = get_ffmpeg_executable()
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    filter_str = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA=11"
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_audio_source),
        "-vn",
        "-af", filter_str,
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        str(output_audio_path),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and output_audio_path.exists() and output_audio_path.stat().st_size > 0


def mux_media(
    video_path: str | Path,
    audio_path: Optional[str | Path],
    output_path: str | Path,
    subtitles_path: Optional[str | Path] = None,
    crf: int = 18,
    preset: str = "medium",
) -> bool:
    """Multiplexes video stream, audio track, and optional subtitles into a final broadcast MP4."""
    ffmpeg_bin = get_ffmpeg_executable()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [ffmpeg_bin, "-y", "-i", str(video_path)]

    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", str(audio_path)])

    # Subtitle handling (soft subtitles via mov_text in MP4)
    if subtitles_path and Path(subtitles_path).exists():
        cmd.extend(["-i", str(subtitles_path)])
        cmd.extend(["-c:s", "mov_text"])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", preset,
    ])

    if audio_path and Path(audio_path).exists():
        cmd.extend([
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-shortest",
        ])

    cmd.append(str(output_path))

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
