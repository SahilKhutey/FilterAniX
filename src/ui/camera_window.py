"""Live Webcam Window with Real-Time Preview & MP4 Recording."""
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QMessageBox,
    QWidget,
)

from src.ui.video_widget import VideoWidget


class LiveCameraWindow(QDialog):
    """Live camera capture, preview, and recording dialog."""

    recording_saved = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Animated Creator — Live Camera")
        self.setMinimumSize(720, 520)
        self.setStyleSheet("""
            QDialog { background-color: #1A1D21; color: #FFFFFF; }
            QPushButton {
                background-color: #2D333B;
                color: #FFFFFF;
                border: 1px solid #444C56;
                padding: 8px 16px;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #373E47; }
            QPushButton:pressed { background-color: #22272E; }
            QPushButton#record_btn { background-color: #D9383A; border-color: #B22B2C; font-weight: bold; }
            QPushButton#record_btn:hover { background-color: #E5484A; }
            QComboBox {
                background-color: #2D333B;
                color: #FFFFFF;
                border: 1px solid #444C56;
                padding: 6px 12px;
                border-radius: 6px;
            }
        """)

        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.is_recording = False
        self.record_start_time = 0.0
        self.output_record_path = Path("samples/recorded_webcam.mp4")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self._init_ui()
        self._start_camera(0)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Controls Header
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Select Camera:"))

        self.cam_selector = QComboBox()
        self.cam_selector.addItem("Camera 0 (Default)", 0)
        self.cam_selector.addItem("Camera 1", 1)
        self.cam_selector.currentIndexChanged.connect(self._on_camera_changed)
        top_bar.addWidget(self.cam_selector)

        top_bar.addStretch()

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #8B949E; font-size: 12px;")
        top_bar.addWidget(self.fps_label)

        layout.addLayout(top_bar)

        # Camera Display Widget
        self.video_widget = VideoWidget(title="Live Camera Feed")
        layout.addWidget(self.video_widget, stretch=1)

        # Recording Status and Buttons
        bottom_bar = QHBoxLayout()

        self.record_btn = QPushButton("🔴 Start Recording")
        self.record_btn.setObjectName("record_btn")
        self.record_btn.clicked.connect(self._toggle_recording)
        bottom_bar.addWidget(self.record_btn)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #8B949E; font-size: 13px;")
        bottom_bar.addWidget(self.status_label)

        bottom_bar.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        bottom_bar.addWidget(self.close_btn)

        layout.addLayout(bottom_bar)

    def _start_camera(self, index: int):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            self.status_label.setText(f"Failed to open Camera {index}")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(33)  # ~30 FPS
        self.status_label.setText(f"Camera {index} active")

    def _on_camera_changed(self, idx: int):
        cam_id = self.cam_selector.currentData()
        self._start_camera(cam_id)

    def _update_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame_bgr = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.video_widget.set_frame(frame_rgb)

        if self.is_recording and self.video_writer is not None:
            self.video_writer.write(frame_bgr)
            elapsed = time.time() - self.record_start_time
            self.status_label.setText(f"Recording: {elapsed:.1f}s | Output: {self.output_record_path.name}")

    def _toggle_recording(self):
        if not self.is_recording:
            # Start recording
            if self.cap is None or not self.cap.isOpened():
                QMessageBox.warning(self, "Camera Error", "No active camera available to record.")
                return

            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0

            self.output_record_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(str(self.output_record_path), fourcc, fps, (w, h))

            self.is_recording = True
            self.record_start_time = time.time()
            self.record_btn.setText("⏹ Stop Recording")
            self.status_label.setText("Recording started...")
        else:
            # Stop recording
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None

            self.record_btn.setText("🔴 Start Recording")
            self.status_label.setText(f"Saved recording to: {self.output_record_path}")
            self.recording_saved.emit(str(self.output_record_path.resolve()))
            QMessageBox.information(
                self, "Recording Complete", f"Webcam footage saved to:\n{self.output_record_path.resolve()}"
            )

    def closeEvent(self, event):
        self.timer.stop()
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()
        if self.cap is not None:
            self.cap.release()
        event.accept()
