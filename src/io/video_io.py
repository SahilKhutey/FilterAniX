"""Video I/O, Metadata Inspection, and FFmpeg Audio-Video Multiplexing."""
from pathlib import Path
from typing import Optional
import cv2

from src.core.models import VideoMetadata
from src.media.ffmpeg import (
    get_ffmpeg_executable,
    get_ffprobe_executable,
    require_ffmpeg,
    require_ffprobe,
    inspect_media,
    extract_audio,
    mux_media,
    run_ffmpeg,
    MediaDetails,
)


def inspect_video(video_path: str | Path) -> VideoMetadata:
    """Inspects a video file and returns its resolution, FPS, frame count, duration, and audio presence."""
    details: MediaDetails = inspect_media(video_path)
    
    # Calculate frame count if needed
    frame_count = 0
    if details.video_duration > 0 and details.fps > 0:
        frame_count = int(round(details.video_duration * details.fps))
    else:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

    return VideoMetadata(
        path=details.path,
        width=details.width,
        height=details.height,
        fps=details.fps,
        frame_count=frame_count,
        duration_sec=details.video_duration,
        has_audio=details.has_audio,
        video_codec=details.video_codec,
        audio_codec=details.audio_codec,
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


def merge_audio_and_video(
    silent_video_path: str | Path,
    audio_source_path: Optional[str | Path],
    final_output_path: str | Path,
    has_audio: bool = True,
    crf: int = 18,
    preset: str = "medium",
) -> bool:
    """Combines processed silent video frames with the preserved original audio track into final high-quality MP4."""
    audio_p = audio_source_path if has_audio else None
    return mux_media(
        video_path=silent_video_path,
        audio_path=audio_p,
        output_path=final_output_path,
        crf=crf,
        preset=preset,
    )

