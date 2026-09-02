"""Lip-Sync Mouth State and Viseme Analyzer."""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np

from src.vision.models import FrameVisionData, FaceData


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
        thresh_closed: float = 0.08,
        thresh_open: float = 0.22,
        thresh_wide: float = 0.45,
    ):
        self.thresh_closed = thresh_closed
        self.thresh_open = thresh_open
        self.thresh_wide = thresh_wide

    def classify_ratio(self, ratio: float) -> VisemeState:
        """Maps mouth opening ratio to VisemeState."""
        if ratio < self.thresh_closed:
            return VisemeState.CLOSED
        elif ratio < self.thresh_open:
            return VisemeState.SLIGHTLY_OPEN
        elif ratio < self.thresh_wide:
            return VisemeState.OPEN
        else:
            return VisemeState.WIDE_OPEN

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
