from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class Point:
    x: float
    y: float
    z: float | None = None
    visibility: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "visibility": self.visibility,
        }


@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    confidence: float | None = None
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "score": self.score,
        }



@dataclass
class Detection:
    label: str
    class_id: int
    confidence: float
    box: BoundingBox



@dataclass
class FaceObservation:
    detected: bool = False
    confidence: float = 0.0
    bbox: BoundingBox | None = None

    landmarks: list[Point] = field(default_factory=list)

    eye_state: str | None = None
    mouth_state: str | None = None


@dataclass
class HandObservation:
    handedness: str | None = None
    confidence: float = 0.0
    landmarks: list[Point] = field(default_factory=list)
    bbox: BoundingBox | None = None


@dataclass
class PoseObservation:
    detected: bool = False
    confidence: float = 0.0
    landmarks: list[Point] = field(default_factory=list)


@dataclass
class MotionObservation:
    score: float = 0.0
    dx: float = 0.0
    dy: float = 0.0


@dataclass
class VisionFrame:
    frame_index: int
    timestamp: float

    width: int
    height: int

    face: FaceObservation = field(
        default_factory=FaceObservation
    )

    pose: PoseObservation = field(
        default_factory=PoseObservation
    )

    hands: list[HandObservation] = field(
        default_factory=list
    )

    motion: MotionObservation = field(
        default_factory=MotionObservation
    )

    scene_id: int = 0
    scene_cut: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,

            "face": {
                "detected": self.face.detected,
                "confidence": self.face.confidence,
                "bbox": (
                    vars(self.face.bbox)
                    if self.face.bbox
                    else None
                ),
                "landmarks": [
                    vars(p)
                    for p in self.face.landmarks
                ],
                "eye_state": self.face.eye_state,
                "mouth_state": self.face.mouth_state,
            },

            "pose": {
                "detected": self.pose.detected,
                "confidence": self.pose.confidence,
                "landmarks": [
                    vars(p)
                    for p in self.pose.landmarks
                ],
            },

            "hands": [
                {
                    "handedness": hand.handedness,
                    "confidence": hand.confidence,
                    "bbox": (
                        vars(hand.bbox)
                        if hand.bbox
                        else None
                    ),
                    "landmarks": [
                        vars(p)
                        for p in hand.landmarks
                    ],
                }
                for hand in self.hands
            ],

            "motion": vars(self.motion),

            "scene_id": self.scene_id,
            "scene_cut": self.scene_cut,

            "metadata": self.metadata,
        }

    @classmethod
    def from_frame_vision_data(cls, data: Any) -> VisionFrame:
        """Converts FrameVisionData into canonical VisionFrame representation."""
        face_obs = FaceObservation()
        if hasattr(data, "faces") and data.faces:
            f0 = data.faces[0]
            face_obs.detected = True
            face_obs.confidence = 0.95
            if hasattr(f0, "bbox") and f0.bbox:
                face_obs.bbox = BoundingBox(
                    x=f0.bbox.x,
                    y=f0.bbox.y,
                    width=f0.bbox.width,
                    height=f0.bbox.height,
                    confidence=1.0,
                )
            if hasattr(f0, "landmarks") and f0.landmarks:
                face_obs.landmarks = [
                    Point(x=lm.x, y=lm.y, z=getattr(lm, "z", None), visibility=getattr(lm, "visibility", None))
                    for lm in f0.landmarks
                ]
            ear = (getattr(f0, "left_eye_ear", 0.0) + getattr(f0, "right_eye_ear", 0.0)) / 2.0
            face_obs.eye_state = "open" if ear > 0.15 else "closed"
            face_obs.mouth_state = "open" if getattr(f0, "mouth_opening", 0.0) > 0.10 else "closed"

        pose_obs = PoseObservation()
        if hasattr(data, "pose") and data.pose:
            p = data.pose
            pose_obs.detected = True
            pose_obs.confidence = 0.90
            if hasattr(p, "landmarks") and p.landmarks:
                pose_obs.landmarks = [
                    Point(x=lm.x, y=lm.y, z=getattr(lm, "z", None), visibility=getattr(lm, "visibility", None))
                    for lm in p.landmarks
                ]

        hands_obs = []
        if hasattr(data, "hands") and data.hands:
            for h in data.hands:
                h_bbox = None
                if hasattr(h, "bbox") and h.bbox:
                    h_bbox = BoundingBox(x=h.bbox.x, y=h.bbox.y, width=h.bbox.width, height=h.bbox.height, confidence=getattr(h, "confidence", 1.0))
                h_lms = []
                if hasattr(h, "landmarks") and h.landmarks:
                    h_lms = [
                        Point(x=lm.x, y=lm.y, z=getattr(lm, "z", None), visibility=getattr(lm, "visibility", None))
                        for lm in h.landmarks
                    ]
                hands_obs.append(
                    HandObservation(
                        handedness=getattr(h, "label", None),
                        confidence=getattr(h, "confidence", 0.0),
                        landmarks=h_lms,
                        bbox=h_bbox,
                    )
                )

        motion_obs = MotionObservation()
        if hasattr(data, "motion") and data.motion:
            m = data.motion
            motion_obs.score = getattr(m, "mean_magnitude", 0.0)
            angle = getattr(m, "mean_angle", 0.0)
            motion_obs.dx = motion_obs.score * math.cos(angle)
            motion_obs.dy = motion_obs.score * math.sin(angle)

        return cls(
            frame_index=data.frame_index,
            timestamp=data.timestamp,
            width=data.width,
            height=data.height,
            face=face_obs,
            pose=pose_obs,
            hands=hands_obs,
            motion=motion_obs,
        )


# Compatibility Aliases
Point3D = Point
from src.vision.models import FrameVisionData as FrameVision
MotionData = MotionObservation

