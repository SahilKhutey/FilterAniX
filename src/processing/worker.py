"""PyQt Worker Thread for Asynchronous, Responsive Video Processing."""
import traceback
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QObject, QThread, pyqtSignal
import numpy as np

from src.core.models import ProcessingProgress
from src.processing.pipeline import VideoPipeline, FrameProcessor


class VideoProcessingWorker(QThread):
    """Background worker thread executing the video pipeline without freezing the GUI."""

    progress_updated = pyqtSignal(object)              # ProcessingProgress
    frame_preview_ready = pyqtSignal(object, object)    # in_rgb (H, W, 3), out_rgb (H, W, 3)
    processing_finished = pyqtSignal(str)              # output_path
    processing_error = pyqtSignal(str)                 # error message string
    status_changed = pyqtSignal(str)                   # short status message

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        frame_processor: Optional[FrameProcessor] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.input_path = str(input_path)
        self.output_path = str(output_path)
        self.frame_processor = frame_processor or FrameProcessor()
        self.pipeline = VideoPipeline(frame_processor=self.frame_processor)
        self._is_cancelled = False

    def cancel(self):
        """Signals the worker thread to safely cancel processing."""
        self._is_cancelled = True
        self.status_changed.emit("Cancelling processing...")

    def run(self):
        """Worker thread entry point."""
        try:
            self.status_changed.emit("Starting video pipeline...")
            
            final_output = self.pipeline.process_video(
                input_path=self.input_path,
                output_path=self.output_path,
                progress_callback=lambda p: self.progress_updated.emit(p),
                frame_callback=lambda in_f, out_f: self.frame_preview_ready.emit(in_f, out_f),
                is_cancelled=lambda: self._is_cancelled,
            )

            if not self._is_cancelled:
                self.processing_finished.emit(final_output)

        except InterruptedError:
            self.status_changed.emit("Processing cancelled.")
        except Exception as e:
            error_trace = traceback.format_exc()
            self.processing_error.emit(f"Error: {str(e)}\n{error_trace}")
