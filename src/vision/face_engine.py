"""Face Analysis Engine using MediaPipe Face Mesh."""
import math
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.vision.models import FaceData, Landmark, BoundingBox


class FaceEngine:
    """Extracts 468/478 dense facial landmarks, facial bounding box, eye blink, and mouth metrics."""

    def __init__(self, max_faces: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.max_faces = max_faces
        self._mesh = None
        try:
            import mediapipe as mp
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self._mesh = None

    def process(self, rgb: np.ndarray) -> List[FaceData]:
        """Analyzes RGB image and returns list of detected FaceData records."""
        h, w = rgb.shape[:2]
        if self._mesh is None:
            return []

        try:
            results = self._mesh.process(rgb)
            if not results.multi_face_landmarks:
                return []

            faces: List[FaceData] = []
            for face_idx, raw_landmarks in enumerate(results.multi_face_landmarks):
                landmarks: List[Landmark] = []
                xs, ys = [], []

                for lm in raw_landmarks.landmark:
                    norm_x = min(1.0, max(0.0, float(lm.x)))
                    norm_y = min(1.0, max(0.0, float(lm.y)))
                    norm_z = float(lm.z)
                    landmarks.append(Landmark(x=norm_x, y=norm_y, z=norm_z, visibility=1.0))
                    xs.append(norm_x)
                    ys.append(norm_y)

                # Compute Bounding Box
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                # Small padding
                pad_x = (max_x - min_x) * 0.08
                pad_y = (max_y - min_y) * 0.08
                bbox = BoundingBox(
                    x=max(0.0, min_x - pad_x),
                    y=max(0.0, min_y - pad_y),
                    width=min(1.0, (max_x - min_x) + 2 * pad_x),
                    height=min(1.0, (max_y - min_y) + 2 * pad_y),
                )

                # Compute Mouth Opening Ratio (Upper lip 13, Lower lip 14 vs Mouth corners 61, 291)
                mouth_opening = 0.0
                if len(landmarks) >= 300:
                    lip_top = landmarks[13]
                    lip_bot = landmarks[14]
                    mouth_l = landmarks[61]
                    mouth_r = landmarks[291]
                    vert_dist = math.hypot(lip_top.x - lip_bot.x, lip_top.y - lip_bot.y)
                    horiz_dist = max(1e-4, math.hypot(mouth_l.x - mouth_r.x, mouth_l.y - mouth_r.y))
                    mouth_opening = vert_dist / horiz_dist

                # Compute Eye Aspect Ratios (Blink tracking)
                left_ear = 0.0
                right_ear = 0.0
                if len(landmarks) >= 390:
                    # Left Eye: 362 (outer), 263 (inner), 386/374, 380/385
                    p_l_out, p_l_in = landmarks[362], landmarks[263]
                    p_l_t, p_l_b = landmarks[386], landmarks[374]
                    horiz_l = max(1e-4, math.hypot(p_l_out.x - p_l_in.x, p_l_out.y - p_l_in.y))
                    vert_l = math.hypot(p_l_t.x - p_l_b.x, p_l_t.y - p_l_b.y)
                    left_ear = vert_l / horiz_l

                    # Right Eye: 33 (outer), 133 (inner), 159/145, 144/158
                    p_r_out, p_r_in = landmarks[33], landmarks[133]
                    p_r_t, p_r_b = landmarks[159], landmarks[145]
                    horiz_r = max(1e-4, math.hypot(p_r_out.x - p_r_in.x, p_r_out.y - p_r_in.y))
                    vert_r = math.hypot(p_r_t.x - p_r_b.x, p_r_t.y - p_r_b.y)
                    right_ear = vert_r / horiz_r

                faces.append(
                    FaceData(
                        face_id=face_idx,
                        landmarks=landmarks,
                        bbox=bbox,
                        landmark_count=len(landmarks),
                        mouth_opening=mouth_opening,
                        left_eye_ear=left_ear,
                        right_eye_ear=right_ear,
                    )
                )

            return faces
        except Exception:
            return []

    def close(self):
        if self._mesh is not None:
            self._mesh.close()
