"""YouTube Multi-Resolution Exporter (720p, 1080p, 1440p, 2160p)."""
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.io.video_io import get_ffmpeg_executable, inspect_video


@dataclass
class YouTubePreset:
    """Preset resolution and bitrate settings for YouTube export."""
    name: str
    width: int
    height: int
    video_bitrate: str
    audio_bitrate: str
    crf: int = 18


YOUTUBE_PRESETS = {
    "720p": YouTubePreset(name="720p", width=1280, height=720, video_bitrate="5M", audio_bitrate="192k", crf=20),
    "1080p": YouTubePreset(name="1080p", width=1920, height=1080, video_bitrate="10M", audio_bitrate="256k", crf=18),
    "1440p": YouTubePreset(name="1440p", width=2560, height=1440, video_bitrate="16M", audio_bitrate="320k", crf=16),
    "2160p": YouTubePreset(name="2160p", width=3840, height=2160, video_bitrate="35M", audio_bitrate="320k", crf=15),
}


class YouTubeExporter:
    """Exports master composite videos into broadcast-grade YouTube MP4 files."""

    def __init__(self, preset_name: str = "1080p"):
        self.preset = YOUTUBE_PRESETS.get(preset_name, YOUTUBE_PRESETS["1080p"])

    def export(
        self,
        input_master_path: str | Path,
        output_export_path: str | Path,
        custom_preset: Optional[str] = None,
    ) -> str:
        input_p = Path(input_master_path).resolve()
        output_p = Path(output_export_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)

        if not input_p.exists():
            raise FileNotFoundError(f"Master input video not found: {input_p}")

        preset = YOUTUBE_PRESETS.get(custom_preset, self.preset) if custom_preset else self.preset
        ffmpeg_bin = get_ffmpeg_executable()

        scale_filter = f"scale={preset.width}:{preset.height}:force_original_aspect_ratio=decrease,pad={preset.width}:{preset.height}:(ow-iw)/2:(oh-ih)/2"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(input_p),
            "-vf", scale_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(preset.crf),
            "-b:v", preset.video_bitrate,
            "-maxrate", preset.video_bitrate,
            "-bufsize", f"{int(preset.video_bitrate[:-1])*2}M",
            "-preset", "slow",
            "-c:a", "aac",
            "-b:a", preset.audio_bitrate,
            "-ar", "48000",
            "-movflags", "+faststart",
            str(output_p),
        ]

        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not output_p.exists():
            raise RuntimeError(f"FFmpeg YouTube export failed: {res.stderr.decode('utf-8', errors='ignore')}")

        return str(output_p)
