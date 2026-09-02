"""Automated Tests for Phase 1 Video I/O, Metadata, Audio Muxing, and Pass-Through Pipeline."""
import math
import subprocess
from pathlib import Path
import cv2
import numpy as np
import pytest

from src.core.models import VideoMetadata, ProcessingProgress
from src.io.video_io import (
    get_ffmpeg_executable,
    inspect_video,
    create_video_writer,
    extract_audio,
    merge_audio_and_video,
)
from src.processing.pipeline import FrameProcessor, VideoPipeline


def create_synthetic_test_video_with_audio(
    output_path: Path | str, num_frames: int = 30, width: int = 320, height: int = 240, fps: int = 30
) -> Path:
    """Generates a synthetic MP4 video containing both visual content and a synthesized audio tone."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_silent = output_path.parent / f"silent_{output_path.name}"

    # 1. Create silent video frames
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_silent), fourcc, fps, (width, height))

    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = [30 + i * 2, 80 + i, 160]
        # Moving circle
        cx = int(width / 2 + 50 * math.sin(i * 0.2))
        cy = int(height / 2 + 30 * math.cos(i * 0.2))
        cv2.circle(img, (cx, cy), 20, (0, 255, 200), -1)
        cv2.putText(img, f"F:{i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(img)

    writer.release()

    # 2. Synthesize audio track and mux via FFmpeg
    ffmpeg_bin = get_ffmpeg_executable()
    duration_sec = num_frames / fps
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", str(temp_silent),
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration_sec}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    if temp_silent.exists():
        temp_silent.unlink()

    return output_path


def test_ffmpeg_availability():
    """Verifies that FFmpeg is located and operational."""
    ffmpeg_bin = get_ffmpeg_executable()
    assert ffmpeg_bin is not None
    assert Path(ffmpeg_bin).exists()


def test_video_inspection(tmp_path):
    """Verifies metadata extraction on a video with audio."""
    test_video = tmp_path / "test_inspect.mp4"
    create_synthetic_test_video_with_audio(test_video, num_frames=45, width=640, height=360, fps=30)

    meta = inspect_video(test_video)
    assert meta.width == 640
    assert meta.height == 360
    assert abs(meta.fps - 30.0) < 1.0
    assert meta.frame_count >= 40
    assert meta.has_audio is True
    assert meta.resolution_str == "640x360"


def test_frame_processor_pass_through():
    """Verifies that the Phase 1 FrameProcessor acts as a strict pass-through."""
    processor = FrameProcessor()
    sample_frame = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    processed = processor.process_frame(sample_frame)
    np.testing.assert_array_equal(sample_frame, processed)


def test_end_to_end_pipeline_with_audio(tmp_path):
    """Verifies the complete Phase 1 pipeline: decode -> pass-through -> encode -> remux audio."""
    input_video = tmp_path / "input_with_audio.mp4"
    output_video = tmp_path / "output_processed.mp4"

    create_synthetic_test_video_with_audio(input_video, num_frames=30, width=320, height=240, fps=30)
    
    pipeline = VideoPipeline(temp_dir=tmp_path / "temp")
    
    progress_records = []
    def on_progress(p: ProcessingProgress):
        progress_records.append(p.percent)

    final_path = pipeline.process_video(
        input_path=input_video,
        output_path=output_video,
        progress_callback=on_progress,
    )

    assert Path(final_path).exists()
    assert Path(final_path).stat().st_size > 500
    assert len(progress_records) > 0
    assert progress_records[-1] == 100.0

    # Inspect the output video to guarantee audio was preserved
    out_meta = inspect_video(final_path)
    assert out_meta.width == 320
    assert out_meta.height == 240
    assert out_meta.has_audio is True
