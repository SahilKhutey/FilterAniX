from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Protobuf 5+ compatibility shim for MediaPipe symbol_database
try:
    from google.protobuf import symbol_database, message_factory
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        symbol_database.SymbolDatabase.GetPrototype = lambda self, descriptor: message_factory.GetMessageClass(descriptor)
except Exception:
    pass

import mediapipe as mp


from .types import BoundingBox, Point3D


class MediaPipeVision:
    """MediaPipe-backed face, pose, hands and person segmentation engine."""

    def __init__(
        self,
        face_detection_confidence: float = 0.5,
        face_tracking_confidence: float = 0.5,
        pose_detection_confidence: float = 0.5,
        pose_tracking_confidence: float = 0.5,
        hand_detection_confidence: float = 0.5,
        hand_tracking_confidence: float = 0.5,
        enable_segmentation: bool = True,
        hand_interval: int = 1,
    ):
        self.mp_face = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.hand_interval = max(1, hand_interval)

        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=face_detection_confidence,
            min_tracking_confidence=face_tracking_confidence,
        )

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=pose_detection_confidence,
            min_tracking_confidence=pose_tracking_confidence,
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=hand_detection_confidence,
            min_tracking_confidence=hand_tracking_confidence,
        )

        self._cached_hands: List[Dict[str, Any]] = []

    @staticmethod
    def _landmark_list(landmarks) -> List[Dict[str, float]]:
        return [
            Point3D(
                x=float(p.x),
                y=float(p.y),
                z=float(getattr(p, "z", 0.0)),
                visibility=float(getattr(p, "visibility", 1.0)),
            ).to_dict()
            for p in landmarks
        ]

    @staticmethod
    def _bbox_from_landmarks(landmarks) -> BoundingBox:
        xs = [float(p.x) for p in landmarks]
        ys = [float(p.y) for p in landmarks]
        x0, x1 = max(0.0, min(xs)), min(1.0, max(xs))
        y0, y1 = max(0.0, min(ys)), min(1.0, max(ys))
        return BoundingBox(x0, y0, x1 - x0, y1 - y0)

    def process(self, frame_bgr: np.ndarray, frame_index: int = 0) -> Dict[str, Any]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        face_result = self.face.process(rgb)
        pose_result = self.pose.process(rgb)

        faces = []
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                points = self._landmark_list(face_landmarks.landmark)
                faces.append({
                    "landmarks": points,
                    "bbox": self._bbox_from_landmarks(face_landmarks.landmark).to_dict(),
                    "landmark_count": len(points),
                })

        pose = None
        if pose_result.pose_landmarks:
            points = self._landmark_list(pose_result.pose_landmarks.landmark)
            pose = {
                "landmarks": points,
                "landmark_count": len(points),
            }

        if self.hand_interval <= 1 or frame_index % self.hand_interval == 0:
            hand_result = self.hands.process(rgb)
            hands = []
            if hand_result.multi_hand_landmarks:
                handedness = hand_result.multi_handedness or []
                for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                    label = "Unknown"
                    score = 0.0
                    if idx < len(handedness):
                        classification = handedness[idx].classification[0]
                        label = classification.label
                        score = float(classification.score)

                    hands.append({
                        "label": label,
                        "score": score,
                        "landmarks": self._landmark_list(hand_landmarks.landmark),
                        "landmark_count": len(hand_landmarks.landmark),
                    })
            self._cached_hands = hands
        else:
            hands = self._cached_hands

        person_mask = None
        if pose_result.segmentation_mask is not None:
            mask = pose_result.segmentation_mask
            binary = (mask > 0.5).astype(np.uint8)
            person_mask = {
                "threshold": 0.5,
                "coverage": float(binary.mean()),
                "shape": [int(binary.shape[0]), int(binary.shape[1])],
            }

        return {
            "faces": faces,
            "pose": pose,
            "hands": hands,
            "person_mask": person_mask,
        }

    def draw(self, frame_bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        out = frame_bgr.copy()

        # Draw face bounding boxes.
        h, w = out.shape[:2]
        for face in result["faces"]:
            b = face["bbox"]
            x1 = int(b["x"] * w)
            y1 = int(b["y"] * h)
            x2 = int((b["x"] + b["width"]) * w)
            y2 = int((b["y"] + b["height"]) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Sparse landmark visualization keeps preview readable.
            for p in face["landmarks"][::20]:
                cv2.circle(
                    out,
                    (int(p["x"] * w), int(p["y"] * h)),
                    1,
                    (255, 255, 255),
                    -1,
                )

        # Pose.
        if result["pose"]:
            connections = self.mp_pose.POSE_CONNECTIONS
            pts = result["pose"]["landmarks"]
            for a, b in connections:
                pa, pb = pts[a], pts[b]
                cv2.line(
                    out,
                    (int(pa["x"] * w), int(pa["y"] * h)),
                    (int(pb["x"] * w), int(pb["y"] * h)),
                    (200, 200, 200),
                    2,
                )

        # Hands.
        for hand in result["hands"]:
            pts = hand["landmarks"]
            for p in pts:
                cv2.circle(
                    out,
                    (int(p["x"] * w), int(p["y"] * h)),
                    2,
                    (255, 255, 255),
                    -1,
                )

        return out

    def close(self):
        self.face.close()
        self.pose.close()
        self.hands.close()
