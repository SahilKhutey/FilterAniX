from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    score: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class Detection:
    label: str
    class_id: int
    confidence: float
    box: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "box": self.box.to_dict(),
        }


@dataclass
class MotionData:
    mean_magnitude: float = 0.0
    mean_angle: float = 0.0
    moving_pixel_ratio: float = 0.0
    valid: bool = False

    def to_dict(self) -> Dict[str, float | bool]:
        return asdict(self)


@dataclass
class FrameVision:
    frame_index: int
    timestamp: float
    width: int
    height: int
    faces: List[Dict[str, Any]]
    pose: Optional[Dict[str, Any]]
    hands: List[Dict[str, Any]]
    person_mask: Optional[Dict[str, Any]]
    motion: MotionData
    objects: List[Detection]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
            "faces": self.faces,
            "pose": self.pose,
            "hands": self.hands,
            "person_mask": self.person_mask,
            "motion": self.motion.to_dict(),
            "objects": [x.to_dict() for x in self.objects],
        }
