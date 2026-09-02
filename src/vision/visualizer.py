"""High-Quality Annotated Debug Visualizer for Vision & Scene Understanding."""
from typing import List, Optional
import cv2
import numpy as np

from src.vision.models import FrameVisionData, FaceData, PoseData, HandData, ObjectData


class VisionVisualizer:
    """Draws rich, broadcast-quality debug overlays on raw frames."""

    def __init__(self):
        # Color Palette (BGR for OpenCV drawing)
        self.color_face = (255, 200, 50)      # Cyan/Yellow
        self.color_pose = (0, 255, 120)       # Bright Green
        self.color_hand_l = (255, 80, 80)     # Blue/Purple
        self.color_hand_r = (80, 80, 255)     # Red/Orange
        self.color_mask = (255, 120, 255)     # Soft Magenta
        self.color_object = (50, 220, 255)    # Yellow/Gold
        self.color_hud = (20, 20, 20)         # Dark HUD

    def render(
        self,
        frame_rgb: np.ndarray,
        data: FrameVisionData,
        mask_uint8: Optional[np.ndarray] = None,
        flow_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Draws all vision layers onto a copy of the RGB image and returns annotated RGB frame."""
        h, w = frame_rgb.shape[:2]
        canvas_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1. Draw Person Segmentation Contour
        if mask_uint8 is not None:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas_bgr, contours, -1, self.color_mask, 2)

        # 2. Draw Object Detection BBoxes
        for obj in data.objects:
            bx = int(obj.bbox.x * w)
            by = int(obj.bbox.y * h)
            bw = int(obj.bbox.width * w)
            bh = int(obj.bbox.height * h)
            cv2.rectangle(canvas_bgr, (bx, by), (bx + bw, by + bh), self.color_object, 2)
            label_text = f"{obj.label} ({obj.confidence:.2f})"
            cv2.putText(
                canvas_bgr, label_text, (bx + 4, max(18, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_object, 1, cv2.LINE_AA
            )

        # 3. Draw Body Pose Skeleton
        if data.pose and data.pose.landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in data.pose.landmarks]
            # Key connections: shoulders (11-12), arms (11-13-15, 12-14-16), torso (11-23, 12-24, 23-24)
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24)
            ]
            for p1_i, p2_i in connections:
                if p1_i < len(pts) and p2_i < len(pts):
                    if data.pose.landmarks[p1_i].visibility > 0.4 and data.pose.landmarks[p2_i].visibility > 0.4:
                        cv2.line(canvas_bgr, pts[p1_i], pts[p2_i], self.color_pose, 2, cv2.LINE_AA)
            for idx, pt in enumerate(pts[:25]):
                if data.pose.landmarks[idx].visibility > 0.4:
                    cv2.circle(canvas_bgr, pt, 4, (0, 200, 255), -1)

        # 4. Draw Hand Skeletons
        for hand in data.hands:
            hand_color = self.color_hand_l if hand.label == "Left" else self.color_hand_r
            h_pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks]
            # Hand bone connections
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17)              # Palm base
            ]
            for p1_i, p2_i in hand_connections:
                if p1_i < len(h_pts) and p2_i < len(h_pts):
                    cv2.line(canvas_bgr, h_pts[p1_i], h_pts[p2_i], hand_color, 2, cv2.LINE_AA)
            for pt in h_pts:
                cv2.circle(canvas_bgr, pt, 3, (255, 255, 255), -1)

        # 5. Draw Face Mesh Points / BBox
        for face in data.faces:
            bx = int(face.bbox.x * w)
            by = int(face.bbox.y * h)
            bw = int(face.bbox.width * w)
            bh = int(face.bbox.height * h)
            cv2.rectangle(canvas_bgr, (bx, by), (bx + bw, by + bh), self.color_face, 1)

            # Subsample face points for clean display
            for idx in range(0, len(face.landmarks), 8):
                lm = face.landmarks[idx]
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(canvas_bgr, (px, py), 1, self.color_face, -1)

        # 6. Draw HUD Telemetry Banner
        hud_h = 36
        overlay = canvas_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (15, 18, 24), -1)
        cv2.addWeighted(overlay, 0.85, canvas_bgr, 0.15, 0, canvas_bgr)

        hud_text = (
            f"F:{data.frame_index:04d} | T:{data.timestamp:.2f}s | "
            f"Faces:{len(data.faces)} | Pose:{'Yes' if data.pose else 'No'} | "
            f"Hands:{len(data.hands)} | Mask:{data.person_mask.coverage*100:.1f}% | "
            f"Motion:{data.motion.mean_magnitude:.2f}px"
        )
        cv2.putText(
            canvas_bgr, hud_text, (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 235, 245), 1, cv2.LINE_AA
        )

        return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)
