"""Standardized Data Models for Phase 2 Vision & Scene Understanding."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Landmark:
    """Normalized 3D coordinate point [0.0, 1.0]."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "z": round(self.z, 5),
            "visibility": round(self.visibility, 3),
        }


@dataclass
class BoundingBox:
    """Normalized bounding box in [0.0, 1.0] range."""
    x: float
    y: float
    width: float
    height: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "width": round(self.width, 5),
            "height": round(self.height, 5),
        }


@dataclass
class FaceData:
    """Extracted facial structure."""
    face_id: int
    landmarks: List[Landmark]
    bbox: BoundingBox
    landmark_count: int
    mouth_opening: float = 0.0
    left_eye_ear: float = 0.0
    right_eye_ear: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "face_id": self.face_id,
            "landmark_count": self.landmark_count,
            "bbox": self.bbox.to_dict(),
            "mouth_opening": round(self.mouth_opening, 4),
            "left_eye_ear": round(self.left_eye_ear, 4),
            "right_eye_ear": round(self.right_eye_ear, 4),
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }


@dataclass
class PoseData:
    """Extracted body skeletal pose."""
    landmarks: List[Landmark]
    bbox: BoundingBox
    landmark_count: int
    torso_center: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "landmark_count": self.landmark_count,
            "bbox": self.bbox.to_dict(),
            "torso_center": [round(c, 5) for c in self.torso_center] if self.torso_center else None,
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }


@dataclass
class HandData:
    """Extracted hand keypoints & handedness."""
    label: str  # 'Left' or 'Right'
    confidence: float
    landmarks: List[Landmark]
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
            "landmarks": [lm.to_dict() for lm in self.landmarks],
        }


@dataclass
class PersonMaskData:
    """Segmentation coverage metadata."""
    threshold: float = 0.5
    coverage: float = 0.0
    bbox: Optional[BoundingBox] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": round(self.threshold, 2),
            "coverage": round(self.coverage, 4),
            "bbox": self.bbox.to_dict() if self.bbox else None,
        }


@dataclass
class MotionData:
    """Optical flow motion telemetry."""
    mean_magnitude: float = 0.0
    mean_angle: float = 0.0
    moving_pixel_ratio: float = 0.0
    valid: bool = False
    dx: float = 0.0
    dy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_magnitude": round(self.mean_magnitude, 4),
            "mean_angle": round(self.mean_angle, 2),
            "moving_pixel_ratio": round(self.moving_pixel_ratio, 4),
            "valid": self.valid,
            "dx": round(self.dx, 4),
            "dy": round(self.dy, 4),
        }


@dataclass
class ObjectData:
    """Detected scene object."""
    label: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class FrameVisionData:
    """Master structured frame record for vision.jsonl."""
    frame_index: int
    timestamp: float
    width: int
    height: int
    faces: List[FaceData] = field(default_factory=list)
    pose: Optional[PoseData] = None
    hands: List[HandData] = field(default_factory=list)
    person_mask: Optional[PersonMaskData] = None
    motion: MotionData = field(default_factory=MotionData)
    objects: List[ObjectData] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "width": self.width,
            "height": self.height,
            "faces": [f.to_dict() for f in self.faces],
            "pose": self.pose.to_dict() if self.pose else None,
            "hands": [h.to_dict() for h in self.hands],
            "person_mask": self.person_mask.to_dict() if self.person_mask else None,
            "motion": self.motion.to_dict(),
            "objects": [obj.to_dict() for obj in self.objects],
        }

    def to_canonical(self):
        from src.vision.types import VisionFrame
        return VisionFrame.from_frame_vision_data(self)
