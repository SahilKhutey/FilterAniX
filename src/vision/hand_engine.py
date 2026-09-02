"""Hand Tracking Engine using MediaPipe Hands."""
from typing import List, Optional
import numpy as np

from src.vision.models import HandData, Landmark, BoundingBox


class HandEngine:
    """Tracks up to 2 hands, classifying left/right orientation and extracting 21 3D joint landmarks per hand."""

    def __init__(self, max_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.max_hands = max_hands
        self._hands = None
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        except Exception:
            self._hands = None

    def process(self, rgb: np.ndarray) -> List[HandData]:
        """Analyzes RGB image and returns list of detected HandData records."""
        if self._hands is None:
            return []

        try:
            results = self._hands.process(rgb)
            if not results.multi_hand_landmarks:
                return []

            hands: List[HandData] = []
            for hand_idx, raw_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine handedness label & score
                label = "Unknown"
                confidence = 0.90
                if results.multi_handedness and hand_idx < len(results.multi_handedness):
                    handedness_info = results.multi_handedness[hand_idx].classification[0]
                    label = handedness_info.label  # 'Left' or 'Right'
                    confidence = float(handedness_info.score)

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
                pad_x = (max_x - min_x) * 0.1
                pad_y = (max_y - min_y) * 0.1
                bbox = BoundingBox(
                    x=max(0.0, min_x - pad_x),
                    y=max(0.0, min_y - pad_y),
                    width=min(1.0, (max_x - min_x) + 2 * pad_x),
                    height=min(1.0, (max_y - min_y) + 2 * pad_y),
                )

                hands.append(
                    HandData(
                        label=label,
                        confidence=confidence,
                        landmarks=landmarks,
                        bbox=bbox,
                    )
                )

            return hands
        except Exception:
            return []

    def close(self):
        if self._hands is not None:
            self._hands.close()
