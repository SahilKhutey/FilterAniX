from pathlib import Path
from typing import List

from .types import BoundingBox, Detection


class NullObjectDetector:
    def detect(self, frame) -> List[Detection]:
        return []


class YOLOObjectDetector:
    """Optional Ultralytics detector. Core Phase 2 does not require it."""

    def __init__(self, model_path: str, confidence: float = 0.35):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        detections = []
        for result in results:
            names = result.names
            if result.boxes is None:
                continue

            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                class_id = int(box.cls[0])

                h, w = frame.shape[:2]
                x1, y1, x2, y2 = xyxy

                detections.append(
                    Detection(
                        label=str(names[class_id]),
                        class_id=class_id,
                        confidence=conf,
                        box=BoundingBox(
                            x=float(x1 / w),
                            y=float(y1 / h),
                            width=float((x2 - x1) / w),
                            height=float((y2 - y1) / h),
                            score=conf,
                        ),
                    )
                )

        return detections
