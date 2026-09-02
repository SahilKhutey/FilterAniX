"""Video I/O, Metadata Inspection, and FFmpeg Audio-Video Multiplexing."""
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from src.core.models import VideoMetadata


def get_ffmpeg_executable() -> str:
    """Finds FFmpeg executable on the system or falls back to imageio-ffmpeg bundled binary."""
    # 1. System PATH
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
        
    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    raise RuntimeError("FFmpeg executable could not be located on system or through imageio-ffmpeg.")


def get_ffprobe_executable() -> Optional[str]:
    """Finds FFprobe executable on the system if present."""
    return shutil.which("ffprobe")


def inspect_video(video_path: str | Path) -> VideoMetadata:
    """Inspects a video file and returns its resolution, FPS, frame count, duration, and audio presence."""
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # 1. Inspect visual parameters via OpenCV
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file via OpenCV: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0.0
    cap.release()

    # 2. Inspect audio presence and codecs via FFmpeg
    has_audio = False
    video_codec = "h264"
    audio_codec = "none"

    ffmpeg_bin = get_ffmpeg_executable()
    try:
        # Run ffmpeg -i on file to capture header info
        proc = subprocess.run(
            [ffmpeg_bin, "-i", str(video_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
        output_text = (proc.stdout or "") + "\n" + (proc.stderr or "")

        # Look for Stream #0:x: Audio
        if re.search(r"Stream #\d+:\d+.*Audio:", output_text, re.IGNORECASE):
            has_audio = True
            audio_match = re.search(r"Stream #\d+:\d+.*Audio:\s*(\w+)", output_text, re.IGNORECASE)
            if audio_match:
                audio_codec = audio_match.group(1)

        video_match = re.search(r"Stream #\d+:\d+.*Video:\s*(\w+)", output_text, re.IGNORECASE)
        if video_match:
            video_codec = video_match.group(1)

        # Fallback duration from FFmpeg if OpenCV returned 0
        if duration_sec <= 0:
            dur_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", output_text)
            if dur_match:
                h, m, s = map(float, dur_match.groups())
                duration_sec = h * 3600 + m * 60 + s
                if frame_count <= 0 and fps > 0:
                    frame_count = int(duration_sec * fps)
    except Exception as e:
        # In case of any probing error, fall back to basic opencv stats
        pass

    return VideoMetadata(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
        has_audio=has_audio,
        video_codec=video_codec,
        audio_codec=audio_codec,
    )


def create_video_writer(
    output_path: str | Path,
    width: int,
    height: int,
    fps: float,
    fourcc_str: str = "mp4v",
) -> cv2.VideoWriter:
    """Creates an OpenCV VideoWriter for intermediate frame sequences."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Failed to open OpenCV VideoWriter for path: {output_path}")
    return writer


def extract_audio(input_video_path: str | Path, output_audio_path: str | Path) -> bool:
    """Extracts raw or AAC audio from the input video container."""
    ffmpeg_bin = get_ffmpeg_executable()
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_video_path),
        "-vn",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_audio_path),
    ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return res.returncode == 0 and output_audio_path.exists() and output_audio_path.stat().st_size > 0


def merge_audio_and_video(
    silent_video_path: str | Path,
    audio_source_path: Optional[str | Path],
    final_output_path: str | Path,
    has_audio: bool = True,
    crf: int = 18,
    preset: str = "medium",
) -> bool:
    """Combines processed silent video frames with the preserved original audio track into final high-quality MP4."""
    ffmpeg_bin = get_ffmpeg_executable()
    final_output_path = Path(final_output_path).resolve()
    final_output_path.parent.mkdir(parents=True, exist_ok=True)

    if has_audio and audio_source_path and Path(audio_source_path).exists():
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(silent_video_path),
            "-i", str(audio_source_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(final_output_path),
        ]
    else:
        # Video only
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(silent_video_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            str(final_output_path),
        ]

    res = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )

    if res.returncode != 0:
        # Fallback to stream copy if libx264 had an issue
        copy_cmd = [
            ffmpeg_bin,
            "-y",
            "-i", str(silent_video_path),
            str(final_output_path)
        ]
        res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return final_output_path.exists() and final_output_path.stat().st_size > 0
