"""Structural Control Map Preprocessor (Edge, Pose, Face, Hand Guides)."""
from typing import List, Optional
import cv2
import numpy as np

from src.art.types import ControlMap, StylePreset
from src.vision.models import FrameVisionData, FaceData, PoseData, HandData


class ControlBuilder:
    """Extracts structural edge, skeletal pose, face mesh, and hand control maps."""

    def __init__(self, style_preset: Optional[StylePreset] = None):
        self.style = style_preset or StylePreset()

    def extract_edge_map(self, rgb: np.ndarray) -> np.ndarray:
        """Extracts structural Canny edge map preserving desk, objects, and silhouette."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.style.edge_canny_low, self.style.edge_canny_high)
        # Dilate slightly for solid structural control
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        edges = cv2.dilate(edges, kernel, iterations=1)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def extract_pose_map(self, height: int, width: int, pose: Optional[PoseData]) -> np.ndarray:
        """Generates OpenPose-style skeletal bone lines on a black canvas."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if not pose or not pose.landmarks:
            return canvas

        pts = [(int(lm.x * width), int(lm.y * height)) for lm in pose.landmarks]
        connections = [
            (11, 12, (255, 0, 0)),    # Shoulders (Blue)
            (11, 13, (255, 128, 0)),  # Left upper arm (Cyan)
            (13, 15, (255, 255, 0)),  # Left forearm (Yellow)
            (12, 14, (0, 255, 0)),    # Right upper arm (Green)
            (14, 16, (0, 255, 255)),  # Right forearm (Orange)
            (11, 23, (128, 0, 255)),  # Left torso (Purple)
            (12, 24, (255, 0, 255)),  # Right torso (Magenta)
            (23, 24, (0, 0, 255)),    # Hips (Red)
        ]

        for p1_i, p2_i, color in connections:
            if p1_i < len(pts) and p2_i < len(pts):
                if pose.landmarks[p1_i].visibility > 0.4 and pose.landmarks[p2_i].visibility > 0.4:
                    cv2.line(canvas, pts[p1_i], pts[p2_i], color, 4, cv2.LINE_AA)

        for idx, pt in enumerate(pts[:25]):
            if pose.landmarks[idx].visibility > 0.4:
                cv2.circle(canvas, pt, 5, (255, 255, 255), -1)

        return canvas

    def extract_face_map(self, height: int, width: int, faces: List[FaceData]) -> np.ndarray:
        """Generates facial mesh contour guides (jaw, eyes, nose, lips)."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for face in faces:
            if not face.landmarks:
                continue
            pts = [(int(lm.x * width), int(lm.y * height)) for lm in face.landmarks]
            # Draw facial key contours if sufficient landmarks exist
            if len(pts) >= 468:
                # Jawline (0-16 approximation), Lips (13, 14, 61, 291), Eyes
                for i in range(0, 468, 6):
                    cv2.circle(canvas, pts[i], 1, (0, 255, 200), -1)
            else:
                # Bounding box fallback
                bx = int(face.bbox.x * width)
                by = int(face.bbox.y * height)
                bw = int(face.bbox.width * width)
                bh = int(face.bbox.height * height)
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (0, 255, 200), 2)
        return canvas

    def extract_hand_map(self, height: int, width: int, hands: List[HandData]) -> np.ndarray:
        """Generates hand joint and bone structure maps."""
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for hand in hands:
            if not hand.landmarks:
                continue
            h_pts = [(int(lm.x * width), int(lm.y * height)) for lm in hand.landmarks]
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),
                (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20),
                (5, 9), (9, 13), (13, 17)
            ]
            for p1, p2 in connections:
                if p1 < len(h_pts) and p2 < len(h_pts):
                    cv2.line(canvas, h_pts[p1], h_pts[p2], (0, 200, 255), 2, cv2.LINE_AA)
            for pt in h_pts:
                cv2.circle(canvas, pt, 3, (255, 255, 255), -1)
        return canvas

    def build_control_map(self, rgb: np.ndarray, vision_data: Optional[FrameVisionData] = None) -> ControlMap:
        """Constructs complete ControlMap structure combining edges, pose, face, and hands."""
        h, w = rgb.shape[:2]
        edge_map = self.extract_edge_map(rgb)
        
        pose_map = None
        face_map = None
        hand_map = None

        if vision_data:
            pose_map = self.extract_pose_map(h, w, vision_data.pose)
            face_map = self.extract_face_map(h, w, vision_data.faces)
            hand_map = self.extract_hand_map(h, w, vision_data.hands)

        # Composite combined control image
        combined = edge_map.copy()
        if pose_map is not None:
            combined = cv2.add(combined, pose_map)
        if face_map is not None:
            combined = cv2.add(combined, face_map)
        if hand_map is not None:
            combined = cv2.add(combined, hand_map)

        return ControlMap(
            edge_map=edge_map,
            pose_map=pose_map,
            face_mesh_map=face_map,
            hand_map=hand_map,
            combined_control=combined,
        )
