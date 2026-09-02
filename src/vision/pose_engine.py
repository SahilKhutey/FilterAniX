"""Body Pose Analysis Engine using MediaPipe Pose."""
from typing import Optional
import numpy as np

from src.vision.models import PoseData, Landmark, BoundingBox


class PoseEngine:
    """Extracts 33 skeletal body landmarks, torso center, and upper-body bounding box."""

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._pose = None
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=True,
                smooth_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self._pose = None

    def process(self, rgb: np.ndarray) -> Optional[PoseData]:
        """Analyzes RGB image and returns PoseData if a person is detected, else None."""
        if self._pose is None:
            return None

        try:
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                return None

            landmarks = []
            xs, ys = [], []

            for lm in results.pose_landmarks.landmark:
                norm_x = min(1.0, max(0.0, float(lm.x)))
                norm_y = min(1.0, max(0.0, float(lm.y)))
                norm_z = float(lm.z)
                vis = float(getattr(lm, "visibility", 1.0))
                landmarks.append(Landmark(x=norm_x, y=norm_y, z=norm_z, visibility=vis))
                xs.append(norm_x)
                ys.append(norm_y)

            # Compute Bounding Box
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            pad_x = (max_x - min_x) * 0.05
            pad_y = (max_y - min_y) * 0.05
            bbox = BoundingBox(
                x=max(0.0, min_x - pad_x),
                y=max(0.0, min_y - pad_y),
                width=min(1.0, (max_x - min_x) + 2 * pad_x),
                height=min(1.0, (max_y - min_y) + 2 * pad_y),
            )

            # Torso Center (Midpoint between shoulders 11, 12 and hips 23, 24)
            torso_center = None
            if len(landmarks) >= 25:
                l_sh, r_sh = landmarks[11], landmarks[12]
                l_hip, r_hip = landmarks[23], landmarks[24]
                tx = (l_sh.x + r_sh.x + l_hip.x + r_hip.x) / 4.0
                ty = (l_sh.y + r_sh.y + l_hip.y + r_hip.y) / 4.0
                torso_center = [tx, ty]

            return PoseData(
                landmarks=landmarks,
                bbox=bbox,
                landmark_count=len(landmarks),
                torso_center=torso_center,
            )
        except Exception:
            return None

    def close(self):
        if self._pose is not None:
            self._pose.close()
