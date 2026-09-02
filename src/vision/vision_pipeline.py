"""Master Vision Engine Orchestrator."""
from typing import Optional, Tuple
import cv2
import numpy as np

from src.vision.models import FrameVisionData
from src.vision.face_engine import FaceEngine
from src.vision.pose_engine import PoseEngine
from src.vision.hand_engine import HandEngine
from src.vision.segmentation_engine import SegmentationEngine
from src.vision.motion_engine import MotionEngine
from src.vision.object_detector import ObjectDetector
from src.vision.visualizer import VisionVisualizer


class VisionEngine:
    """End-to-end vision understanding engine analyzing face, pose, hands, segmentation, motion, and objects."""

    def __init__(self, enable_objects: bool = True):
        self.face_engine = FaceEngine()
        self.pose_engine = PoseEngine()
        self.hand_engine = HandEngine()
        self.segmentation_engine = SegmentationEngine()
        self.motion_engine = MotionEngine()
        self.object_detector = ObjectDetector() if enable_objects else None
        self.visualizer = VisionVisualizer()

    def process_frame(
        self,
        rgb: np.ndarray,
        frame_index: int = 0,
        timestamp: float = 0.0,
        generate_annotated: bool = True,
    ) -> Tuple[FrameVisionData, Optional[np.ndarray]]:
        """Extracts complete structured vision data and optional annotated image.
        
        Returns:
            vision_data: FrameVisionData dataclass
            annotated_rgb: (H, W, 3) uint8 image or None
        """
        h, w = rgb.shape[:2]

        # 1. Face Analysis
        faces = self.face_engine.process(rgb)

        # 2. Body Pose
        pose = self.pose_engine.process(rgb)

        # 3. Hand Tracking
        hands = self.hand_engine.process(rgb)

        # 4. Person Segmentation
        mask_uint8, person_mask_data = self.segmentation_engine.process(rgb)

        # 5. Optical Flow Motion
        flow_field, motion_data = self.motion_engine.process(rgb)

        # 6. Object Detection
        objects = self.object_detector.process(rgb) if self.object_detector else []

        vision_data = FrameVisionData(
            frame_index=frame_index,
            timestamp=timestamp,
            width=w,
            height=h,
            faces=faces,
            pose=pose,
            hands=hands,
            person_mask=person_mask_data,
            motion=motion_data,
            objects=objects,
        )

        annotated_rgb = None
        if generate_annotated:
            annotated_rgb = self.visualizer.render(
                frame_rgb=rgb,
                data=vision_data,
                mask_uint8=mask_uint8,
                flow_field=flow_field,
            )

        return vision_data, annotated_rgb

    def close(self):
        """Releases underlying MediaPipe and OpenCV resources."""
        self.face_engine.close()
        self.pose_engine.close()
        self.hand_engine.close()
        self.segmentation_engine.close()
