"""Phase 2 Vision Package."""
# Protobuf 5+ compatibility shim for MediaPipe symbol_database
try:
    from google.protobuf import symbol_database, message_factory
    if not hasattr(symbol_database.SymbolDatabase, "GetPrototype"):
        symbol_database.SymbolDatabase.GetPrototype = lambda self, descriptor: message_factory.GetMessageClass(descriptor)
except Exception:
    pass

from src.vision.types import (
    Point,
    BoundingBox,
    FaceObservation,
    HandObservation,
    PoseObservation,
    MotionObservation,
    VisionFrame,
)
from src.vision.schema import validate_vision_frame
from src.vision.jsonl import write_vision_jsonl, read_vision_jsonl
from src.vision.alignment import validate_frame_sequence
from src.vision.utils import clamp_confidence
from src.vision.models import (
    Landmark,
    FaceData,
    PoseData,
    HandData,
    PersonMaskData,
    MotionData,
    ObjectData,
    FrameVisionData,
)
from src.vision.face_engine import FaceEngine
from src.vision.pose_engine import PoseEngine
from src.vision.hand_engine import HandEngine
from src.vision.segmentation_engine import SegmentationEngine
from src.vision.motion_engine import MotionEngine
from src.vision.object_detector import ObjectDetector
from src.vision.visualizer import VisionVisualizer
from src.vision.vision_pipeline import VisionEngine

__all__ = [
    "Point",
    "BoundingBox",
    "FaceObservation",
    "HandObservation",
    "PoseObservation",
    "MotionObservation",
    "VisionFrame",
    "validate_vision_frame",
    "write_vision_jsonl",
    "read_vision_jsonl",
    "validate_frame_sequence",
    "clamp_confidence",
    "Landmark",
    "FaceData",
    "PoseData",
    "HandData",
    "PersonMaskData",
    "MotionData",
    "ObjectData",
    "FrameVisionData",
    "FaceEngine",
    "PoseEngine",
    "HandEngine",
    "SegmentationEngine",
    "MotionEngine",
    "ObjectDetector",
    "VisionVisualizer",
    "VisionEngine",
]
