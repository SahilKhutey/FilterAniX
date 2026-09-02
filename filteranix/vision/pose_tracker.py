"""Facial mesh, skeletal pose, and hand tracking for anime feature guidance."""
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class PoseTracker:
    """Tracks facial landmarks, upper-body pose, and hands for structural rendering guidance."""

    def __init__(self, enable_face: bool = True, enable_pose: bool = True, enable_hands: bool = True):
        self.enable_face = enable_face
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        
        self._mp_holistic = None
        try:
            import mediapipe as mp
            if hasattr(mp.solutions, "holistic"):
                self._mp_holistic = mp.solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    refine_face_landmarks=True,
                )
        except Exception:
            self._mp_holistic = None

    def process(self, rgb: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """Processes RGB image and extracts landmark coordinates.
        
        Returns:
            Dict containing:
                - 'face_landmarks': (468/478, 3) or None
                - 'pose_landmarks': (33, 3) or None
                - 'left_hand_landmarks': (21, 3) or None
                - 'right_hand_landmarks': (21, 3) or None
        """
        h, w = rgb.shape[:2]
        output: Dict[str, Optional[np.ndarray]] = {
            "face_landmarks": None,
            "pose_landmarks": None,
            "left_hand_landmarks": None,
            "right_hand_landmarks": None,
        }

        if self._mp_holistic is not None:
            try:
                results = self._mp_holistic.process(rgb)
                if results.face_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in results.face_landmarks.landmark], dtype=np.float32)
                    output["face_landmarks"] = pts
                if results.pose_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
                    output["pose_landmarks"] = pts
                if results.left_hand_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32)
                    output["left_hand_landmarks"] = pts
                if results.right_hand_landmarks:
                    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32)
                    output["right_hand_landmarks"] = pts
                return output
            except Exception:
                pass

        # OpenCV Haar Cascade / Gradient Fallback for Face
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            fx, fy, fw, fh = faces[0]
            # Synthesize basic face anchor points (eyes, nose, mouth center)
            pts = np.array([
                [fx + fw * 0.3, fy + fh * 0.35, 0.0],  # Left eye
                [fx + fw * 0.7, fy + fh * 0.35, 0.0],  # Right eye
                [fx + fw * 0.5, fy + fh * 0.55, 0.0],  # Nose
                [fx + fw * 0.5, fy + fh * 0.75, 0.0],  # Mouth
                [fx + fw * 0.5, fy + fh, 0.0],         # Chin
            ], dtype=np.float32)
            output["face_landmarks"] = pts

        return output
