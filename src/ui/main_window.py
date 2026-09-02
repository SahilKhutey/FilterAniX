"""Master Phase 1 GUI for Animated Creator."""
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QGridLayout,
    QSplitter,
    QFrame,
)

from src.core.models import VideoMetadata, ProcessingProgress
from src.io.video_io import inspect_video
from src.processing.pipeline import FrameProcessor
from src.processing.worker import VideoProcessingWorker
from src.ui.video_widget import VideoWidget
from src.ui.camera_window import LiveCameraWindow


class MainWindow(QMainWindow):
    """Main application window for Animated Creator - Phase 1."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ANIMATED CREATOR — PHASE 1: FOUNDATION & VIDEO PIPELINE")
        self.resize(1150, 780)
        self.setMinimumSize(900, 600)

        self.current_video_path: Optional[str] = None
        self.current_metadata: Optional[VideoMetadata] = None
        self.output_video_path: Optional[str] = None
        self.worker: Optional[VideoProcessingWorker] = None

        # Playback Preview Timer
        self.preview_cap: Optional[cv2.VideoCapture] = None
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._on_preview_tick)

        self._apply_theme()
        self._init_ui()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0E1117; color: #E6EDF3; }
            QWidget { color: #E6EDF3; font-family: 'Segoe UI', -apple-system, sans-serif; }
            QGroupBox {
                border: 1px solid #30363D;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px;
                font-weight: bold;
                font-size: 13px;
                background-color: #161B22;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #58A6FF; }
            QPushButton {
                background-color: #21262D;
                color: #C9D1D9;
                border: 1px solid #30363D;
                padding: 9px 18px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #30363D; border-color: #8B949E; color: #FFFFFF; }
            QPushButton:pressed { background-color: #161B22; }
            QPushButton#primary_btn {
                background-color: #238636;
                border-color: #2EA043;
                color: #FFFFFF;
            }
            QPushButton#primary_btn:hover { background-color: #2EA043; }
            QPushButton#cancel_btn {
                background-color: #DA3633;
                border-color: #F85149;
                color: #FFFFFF;
            }
            QPushButton#cancel_btn:hover { background-color: #F85149; }
            QProgressBar {
                border: 1px solid #30363D;
                border-radius: 6px;
                text-align: center;
                height: 22px;
                background-color: #161B22;
                font-weight: bold;
                color: #FFFFFF;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1F6FEB, stop:1 #58A6FF);
                border-radius: 5px;
            }
            QLabel { font-size: 13px; }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(18, 16, 18, 16)
        main_layout.setSpacing(14)

        # Header Title Bar
        header = QHBoxLayout()
        title_label = QLabel("🎬 ANIMATED CREATOR — Phase 1: Foundation & Video Pipeline")
        title_label.setStyleSheet("font-size: 17px; font-weight: bold; color: #58A6FF;")
        header.addWidget(title_label)
        header.addStretch()
        
        self.camera_btn = QPushButton("📹 Open Live Camera")
        self.camera_btn.clicked.connect(self._open_camera_window)
        header.addWidget(self.camera_btn)
        main_layout.addLayout(header)

        # Action Buttons Toolbar
        btn_bar = QHBoxLayout()
        self.open_btn = QPushButton("📂 Upload / Open Video")
        self.open_btn.clicked.connect(self._open_video_file)
        btn_bar.addWidget(self.open_btn)

        self.process_btn = QPushButton("⚡ Process Video (Pass-Through)")
        self.process_btn.setObjectName("primary_btn")
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._start_processing)
        btn_bar.addWidget(self.process_btn)

        self.cancel_btn = QPushButton("⏹ Cancel Processing")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_processing)
        btn_bar.addWidget(self.cancel_btn)

        btn_bar.addStretch()
        main_layout.addLayout(btn_bar)

        # Metadata Details Box
        self.meta_group = QGroupBox("Video Metadata & Inspection")
        meta_layout = QGridLayout(self.meta_group)
        self.meta_labels = {
            "Resolution": QLabel("Resolution: --"),
            "FPS": QLabel("FPS: --"),
            "Frames": QLabel("Frames: --"),
            "Duration": QLabel("Duration: --"),
            "Audio": QLabel("Audio: --"),
            "Video Codec": QLabel("Codec: --"),
        }
        r, c = 0, 0
        for key, lbl in self.meta_labels.items():
            lbl.setStyleSheet("color: #8B949E; font-size: 12px;")
            meta_layout.addWidget(lbl, r, c)
            c += 1
            if c >= 3:
                c = 0
                r += 1
        main_layout.addWidget(self.meta_group)

        # Dual Video Previews (Splitter)
        preview_container = QHBoxLayout()
        preview_container.setSpacing(14)

        self.input_preview = VideoWidget(title="Input Video Preview")
        self.output_preview = VideoWidget(title="Output Processed Preview")

        preview_container.addWidget(self.input_preview, stretch=1)
        preview_container.addWidget(self.output_preview, stretch=1)
        main_layout.addLayout(preview_container, stretch=1)

        # Progress & Status Bar
        progress_box = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_box.addWidget(self.progress_bar)

        status_bar = QHBoxLayout()
        self.status_label = QLabel("Status: Ready. Please open a video to begin.")
        self.status_label.setStyleSheet("color: #8B949E; font-weight: 500;")
        status_bar.addWidget(self.status_label)

        status_bar.addStretch()

        self.fps_status = QLabel("")
        self.fps_status.setStyleSheet("color: #58A6FF; font-weight: bold;")
        status_bar.addWidget(self.fps_status)

        progress_box.addLayout(status_bar)
        main_layout.addLayout(progress_box)

    def _open_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*.*)",
        )
        if not file_path:
            return

        self._load_video(file_path)

    def _load_video(self, file_path: str):
        try:
            self.current_video_path = file_path
            self.current_metadata = inspect_video(file_path)

            # Update Metadata UI
            summary = self.current_metadata.summary_dict()
            self.meta_labels["Resolution"].setText(f"Resolution: <b>{summary['Resolution']}</b>")
            self.meta_labels["FPS"].setText(f"FPS: <b>{summary['FPS']}</b>")
            self.meta_labels["Frames"].setText(f"Frames: <b>{summary['Frames']}</b>")
            self.meta_labels["Duration"].setText(f"Duration: <b>{summary['Duration']}</b>")
            self.meta_labels["Audio"].setText(f"Audio Track: <b>{summary['Audio']}</b>")
            self.meta_labels["Video Codec"].setText(f"Codec: <b>{summary['Video Codec']}</b>")

            # Load first frame into input preview
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.input_preview.set_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()

            self.output_preview.clear_frame()
            self.process_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(file_path).name} | Ready to process.")
            self.progress_bar.setValue(0)
        except Exception as e:
            QMessageBox.critical(self, "Video Load Error", f"Failed to inspect video:\n{str(e)}")

    def _start_processing(self):
        if not self.current_video_path:
            return

        input_p = Path(self.current_video_path)
        out_name = f"{input_p.stem}_phase1_processed.mp4"
        output_p = input_p.parent / out_name

        self.output_video_path = str(output_p)
        self.process_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)

        # Launch background worker thread
        self.worker = VideoProcessingWorker(
            input_path=self.current_video_path,
            output_path=self.output_video_path,
            frame_processor=FrameProcessor(),
        )

        self.worker.progress_updated.connect(self._on_progress)
        self.worker.frame_preview_ready.connect(self._on_frame_preview)
        self.worker.status_changed.connect(self._on_status)
        self.worker.processing_finished.connect(self._on_finished)
        self.worker.processing_error.connect(self._on_error)

        self.worker.start()

    def _cancel_processing(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_label.setText("Cancelling pipeline...")

    def _on_progress(self, p: ProcessingProgress):
        self.progress_bar.setValue(int(p.percent))
        self.status_label.setText(
            f"Processing: {p.current_frame}/{p.total_frames} frames ({p.percent:.1f}%) | ETA: {p.eta_sec:.1f}s"
        )
        self.fps_status.setText(f"Speed: {p.fps:.1f} FPS")

    def _on_frame_preview(self, in_rgb: np.ndarray, out_rgb: np.ndarray):
        self.input_preview.set_frame(in_rgb)
        self.output_preview.set_frame(out_rgb)

    def _on_status(self, msg: str):
        self.status_label.setText(f"Status: {msg}")

    def _on_finished(self, out_path: str):
        self.process_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText(f"✅ Processing complete: {Path(out_path).name}")
        self.fps_status.setText("DONE")

        QMessageBox.information(
            self,
            "Phase 1 Complete",
            f"Successfully processed video through the Phase 1 Foundation Pipeline!\n\nOutput: {out_path}",
        )

    def _on_error(self, err: str):
        self.process_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("❌ Processing failed.")
        QMessageBox.critical(self, "Pipeline Error", err)

    def _open_camera_window(self):
        cam_win = LiveCameraWindow(self)
        cam_win.recording_saved.connect(self._load_video)
        cam_win.exec()

    def _on_preview_tick(self):
        pass
