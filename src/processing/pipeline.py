"""Phase 1 Frame Processing & Media Pipeline Orchestration."""
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np

from src.core.models import ProcessingProgress, VideoMetadata
from src.io.video_io import inspect_video, create_video_writer, merge_audio_and_video


class FrameProcessor:
    """The central Phase 1 extension point.
    
    In Phase 1, this acts as a pure pass-through processor, verifying the entire
    codec, container, decode/encode, and audio muxing pipeline before adding AI.
    """

    def __init__(self, add_demo_timestamp: bool = False):
        self.add_demo_timestamp = add_demo_timestamp

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Processes a single BGR/RGB video frame.
        
        Args:
            frame: uint8 image array (H, W, 3)
            
        Returns:
            processed_frame: uint8 image array (H, W, 3)
        """
        if self.add_demo_timestamp:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "PHASE 1 PASS-THROUGH",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )
            return annotated
            
        return frame


class VideoPipeline:
    """Orchestrates decoding, sequential frame-by-frame processing, silent encoding, and audio remuxing."""

    def __init__(
        self,
        frame_processor: Optional[FrameProcessor] = None,
        temp_dir: Optional[str | Path] = None,
        crf: int = 18,
        preset: str = "medium",
    ):
        self.frame_processor = frame_processor or FrameProcessor()
        self.temp_dir = Path(temp_dir or "temp_processing")
        self.crf = crf
        self.preset = preset

    def process_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        frame_callback: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
        is_cancelled: Optional[Callable[[], bool]] = None,
    ) -> str:
        """Executes the complete Phase 1 video pipeline on the specified input video."""
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Input video does not exist: {input_path}")

        # 1. Metadata Inspection
        metadata: VideoMetadata = inspect_video(input_path)
        total_frames = max(1, metadata.frame_count)
        fps = metadata.fps if metadata.fps > 0 else 30.0

        # 2. Prepare temporary silent video file
        temp_silent_video = self.temp_dir / f"silent_temp_{int(time.time()*1000)}.mp4"
        writer = create_video_writer(
            output_path=temp_silent_video,
            width=metadata.width,
            height=metadata.height,
            fps=fps,
            fourcc_str="mp4v",
        )

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video capture for: {input_path}")

        start_time = time.time()
        current_frame_idx = 0

        try:
            while True:
                if is_cancelled and is_cancelled():
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_idx += 1

                # Frame Processing (Pass-Through in Phase 1)
                processed_frame = self.frame_processor.process_frame(frame)

                # Write processed frame to intermediate silent stream
                writer.write(processed_frame)

                # Send preview frame callback (RGB converted for UI)
                if frame_callback and (current_frame_idx % 2 == 0 or current_frame_idx == total_frames):
                    in_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    out_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_callback(in_rgb, out_rgb)

                # Progress calculation
                if progress_callback and (current_frame_idx % 5 == 0 or current_frame_idx == total_frames):
                    now = time.time()
                    elapsed = max(0.001, now - start_time)
                    proc_fps = current_frame_idx / elapsed
                    percent = min(100.0, (current_frame_idx / total_frames) * 100.0)
                    remaining_frames = max(0, total_frames - current_frame_idx)
                    eta = remaining_frames / proc_fps if proc_fps > 0 else 0.0

                    progress = ProcessingProgress(
                        current_frame=current_frame_idx,
                        total_frames=total_frames,
                        percent=percent,
                        fps=proc_fps,
                        elapsed_sec=elapsed,
                        eta_sec=eta,
                        status_message=f"Processing frame {current_frame_idx}/{total_frames} ({percent:.1f}%)",
                    )
                    progress_callback(progress)

        finally:
            cap.release()
            writer.release()

        if is_cancelled and is_cancelled():
            if temp_silent_video.exists():
                temp_silent_video.unlink(missing_ok=True)
            raise InterruptedError("Video processing was cancelled by user.")

        # 3. Audio Preservation & Final FFmpeg Multiplexing
        if progress_callback:
            progress_callback(
                ProcessingProgress(
                    current_frame=current_frame_idx,
                    total_frames=total_frames,
                    percent=100.0,
                    fps=0.0,
                    elapsed_sec=time.time() - start_time,
                    eta_sec=0.0,
                    status_message="Finalizing video & muxing audio with FFmpeg...",
                )
            )

        success = merge_audio_and_video(
            silent_video_path=temp_silent_video,
            audio_source_path=input_path,
            final_output_path=output_path,
            has_audio=metadata.has_audio,
            crf=self.crf,
            preset=self.preset,
        )

        # 4. Clean up temporary files
        if temp_silent_video.exists():
            try:
                temp_silent_video.unlink(missing_ok=True)
            except Exception:
                pass

        if not success or not output_path.exists():
            raise RuntimeError(f"Failed to generate final MP4 with FFmpeg at: {output_path}")

        return str(output_path)
