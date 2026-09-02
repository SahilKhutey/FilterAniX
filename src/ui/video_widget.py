"""High-Performance Video Display Widget for PyQt6."""
from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy


class VideoWidget(QWidget):
    """Renders video frames with aspect ratio preservation and placeholder styling."""

    def __init__(self, title: str = "Preview", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.title = title
        self.current_pixmap: Optional[QPixmap] = None

        self.setMinimumSize(320, 200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background-color: #121417; border: 1px solid #2A2E35; border-radius: 8px;")

    def set_frame(self, frame_rgb: np.ndarray):
        """Sets and displays an RGB uint8 image frame."""
        if frame_rgb is None or frame_rgb.size == 0:
            self.clear_frame()
            return

        h, w, c = frame_rgb.shape
        bytes_per_line = c * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(q_img)
        self.update()

    def clear_frame(self):
        """Clears current image and displays the placeholder."""
        self.current_pixmap = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = self.rect()

        if self.current_pixmap and not self.current_pixmap.isNull():
            # Scale preserving aspect ratio
            scaled_pixmap = self.current_pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (rect.width() - scaled_pixmap.width()) // 2
            y = (rect.height() - scaled_pixmap.height()) // 2
            painter.drawPixmap(x, y, scaled_pixmap)

            # Draw small title badge in upper-left corner
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            painter.drawRoundedRect(x + 10, y + 10, len(self.title) * 8 + 20, 24, 4, 4)
            painter.setPen(QColor(230, 230, 230))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(x + 20, y + 27, self.title)
        else:
            # Draw placeholder
            painter.setPen(QColor(110, 120, 135))
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Medium))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self.title}\n(No Video Frame)")
