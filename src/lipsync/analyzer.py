from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from .timeline import VisemeFrame
from src.vision.models import FrameVisionData, FaceData


def _get_value(data: Dict[str, Any], *keys, default=0.0):
    for key in keys:
        if key in data:
            return data[key]

    return default


def extract_mouth_open(observation: Dict[str, Any]) -> float:
    """
    Supports several possible Phase 2 mouth schemas.
    Returns normalized mouth opening in [0, 1].
    """
    if not observation:
        return 0.0

    value = _get_value(
        observation,
        "mouth_open",
        "mouth_openness",
        "mouth_opening",
        "mouthOpen",
        default=None,
    )

    if value is not None:
        return max(0.0, min(1.0, float(value)))

    landmarks = observation.get("landmarks")

    if not landmarks:
        return 0.0

    if isinstance(landmarks, dict):
        upper = landmarks.get("upper_lip")
        lower = landmarks.get("lower_lip")

        if upper is not None and lower is not None:
            distance = abs(float(lower[1]) - float(upper[1]))
            return max(0.0, min(1.0, distance))

    elif isinstance(landmarks, list) and len(landmarks) >= 2:
        pass

    return 0.0


def classify_mouth(
    openness: float,
    closed_threshold: float = 0.10,
    slightly_open_threshold: float = 0.22,
    open_threshold: float = 0.40,
) -> str:
    if openness < closed_threshold:
        return "closed"

    if openness < slightly_open_threshold:
        return "slightly_open"

    if openness < open_threshold:
        return "open"

    return "wide_open"


def analyze_mouth_frame(
    frame_index: int,
    timestamp: float,
    observation: Optional[Dict[str, Any]],
) -> VisemeFrame:
    openness = extract_mouth_open(observation or {})
    state = classify_mouth(openness)

    return VisemeFrame(
        frame_index=frame_index,
        timestamp=timestamp,
        mouth_open=openness,
        state=state,
    )


class VisemeState(str, Enum):
    CLOSED = "closed"
    SLIGHTLY_OPEN = "slightly_open"
    OPEN = "open"
    WIDE_OPEN = "wide_open"


@dataclass
class LipSyncRecord:
    """Frame-level lip-sync telemetry."""
    frame_index: int
    timestamp: float
    mouth_open_ratio: float
    viseme: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "mouth_open_ratio": round(self.mouth_open_ratio, 4),
            "viseme": self.viseme,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LipSyncRecord":
        return cls(
            frame_index=d.get("frame_index", 0),
            timestamp=float(d.get("timestamp", 0.0)),
            mouth_open_ratio=float(d.get("mouth_open_ratio", 0.0)),
            viseme=d.get("viseme", "closed"),
        )


class LipSyncAnalyzer:
    """Classifies continuous mouth opening ratio into 4 discrete creator-anime viseme states."""

    def __init__(
        self,
        thresh_closed: float = 0.10,
        thresh_open: float = 0.22,
        thresh_wide: float = 0.40,
    ):
        self.thresh_closed = thresh_closed
        self.thresh_open = thresh_open
        self.thresh_wide = thresh_wide

    def classify_ratio(self, ratio: float) -> VisemeState:
        """Maps mouth opening ratio to VisemeState."""
        state_str = classify_mouth(
            ratio,
            closed_threshold=self.thresh_closed,
            slightly_open_threshold=self.thresh_open,
            open_threshold=self.thresh_wide,
        )
        return VisemeState(state_str)

    def analyze_frame(
        self,
        frame_index: int,
        timestamp: float,
        vision_data: Optional[FrameVisionData] = None,
    ) -> LipSyncRecord:
        """Extracts mouth opening from vision data and produces a LipSyncRecord."""
        ratio = 0.0
        if vision_data and vision_data.faces:
            ratio = vision_data.faces[0].mouth_opening

        viseme = self.classify_ratio(ratio)
        return LipSyncRecord(
            frame_index=frame_index,
            timestamp=timestamp,
            mouth_open_ratio=ratio,
            viseme=viseme.value,
        )



def build_lipsync(
    video: str | Any,
    vision_jsonl: str | Any,
    output: str | Any,
):
    from build_lipsync import build_lipsync as _build
    return _build(video, vision_jsonl, output)

