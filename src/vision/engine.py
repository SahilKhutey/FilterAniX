from pathlib import Path
from typing import Optional, Dict, Any
import cv2

from .mediapipe_engine import MediaPipeVision
from .motion import OpticalFlowMotion
from .objects import NullObjectDetector, YOLOObjectDetector
from .types import FrameVision


class VisionEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None, yolo_model=None):
        config = config or {}
        vision_cfg = config.get("vision", {})
        object_cfg = config.get("objects", {})

        self.mp = MediaPipeVision(
            face_detection_confidence=vision_cfg.get("face_detection_confidence", 0.5),
            face_tracking_confidence=vision_cfg.get("face_tracking_confidence", 0.5),
            pose_detection_confidence=vision_cfg.get("pose_detection_confidence", 0.5),
            pose_tracking_confidence=vision_cfg.get("pose_tracking_confidence", 0.5),
            hand_detection_confidence=vision_cfg.get("hand_detection_confidence", 0.5),
            hand_tracking_confidence=vision_cfg.get("hand_tracking_confidence", 0.5),
            enable_segmentation=vision_cfg.get("segmentation", True),
        )

        self.motion = OpticalFlowMotion() if vision_cfg.get("optical_flow", True) else None

        if yolo_model or object_cfg.get("enabled", False):
            model = yolo_model or object_cfg.get("model")
            self.objects = YOLOObjectDetector(
                model,
                confidence=float(object_cfg.get("confidence", 0.35)),
            )
        else:
            self.objects = NullObjectDetector()

    def process_frame(
        self,
        frame,
        frame_index: int,
        timestamp: float,
    ) -> FrameVision:

        analysis = self.mp.process(frame)
        motion = self.motion.process(frame) if self.motion else None

        if motion is None:
            from .types import MotionData
            motion = MotionData(valid=False)

        objects = self.objects.detect(frame)

        h, w = frame.shape[:2]

        return FrameVision(
            frame_index=frame_index,
            timestamp=timestamp,
            width=w,
            height=h,
            faces=analysis["faces"],
            pose=analysis["pose"],
            hands=analysis["hands"],
            person_mask=analysis["person_mask"],
            motion=motion,
            objects=objects,
        )

    def draw_overlay(self, frame, vision: FrameVision):
        analysis = {
            "faces": vision.faces,
            "pose": vision.pose,
            "hands": vision.hands,
        }
        out = self.mp.draw(frame, analysis)

        text = (
            f"frame={vision.frame_index} "
            f"faces={len(vision.faces)} "
            f"hands={len(vision.hands)} "
            f"pose={'yes' if vision.pose else 'no'} "
            f"motion={vision.motion.mean_magnitude:.2f}"
        )

        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            out,
            text,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return out

    def close(self):
        self.mp.close()
