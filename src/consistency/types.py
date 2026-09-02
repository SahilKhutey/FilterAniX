"""Data Types and Schema Models for Phase 4 Consistency & Temporal Planning."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ReferenceProfile:
    """Lightweight visual signature describing canonical character design."""
    name: str = "creator_canonical"
    color_hist: List[float] = field(default_factory=list)
    dominant_palette: List[List[int]] = field(default_factory=list)
    edge_density: float = 0.0
    aspect_ratio: float = 1.0
    mean_lab: List[float] = field(default_factory=lambda: [128.0, 128.0, 128.0])
    std_lab: List[float] = field(default_factory=lambda: [30.0, 20.0, 20.0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "edge_density": round(self.edge_density, 5),
            "aspect_ratio": round(self.aspect_ratio, 4),
            "mean_lab": [round(x, 2) for x in self.mean_lab],
            "std_lab": [round(x, 2) for x in self.std_lab],
            "dominant_palette": self.dominant_palette,
            "color_hist": [round(x, 6) for x in self.color_hist],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReferenceProfile":
        return cls(
            name=data.get("name", "creator_canonical"),
            color_hist=data.get("color_hist", []),
            dominant_palette=data.get("dominant_palette", []),
            edge_density=float(data.get("edge_density", 0.0)),
            aspect_ratio=float(data.get("aspect_ratio", 1.0)),
            mean_lab=data.get("mean_lab", [128.0, 128.0, 128.0]),
            std_lab=data.get("std_lab", [30.0, 20.0, 20.0]),
        )


@dataclass
class TemporalState:
    """Active temporal controller memory tracking state across frames."""
    scene_id: int = 0
    last_keyframe_idx: int = 0
    frames_since_keyframe: int = 0
    stable_count: int = 0
    last_similarity: float = 1.0


@dataclass
class RenderDecision:
    """Structured temporal decision for a single frame."""
    frame_index: int
    timestamp: float
    scene_id: int
    is_scene_cut: bool
    is_keyframe: bool
    motion_score: float
    reference_strength: float
    preserve_previous: bool
    similarity_warning: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 4),
            "scene_id": self.scene_id,
            "is_scene_cut": self.is_scene_cut,
            "is_keyframe": self.is_keyframe,
            "motion_score": round(self.motion_score, 4),
            "reference_strength": round(self.reference_strength, 3),
            "preserve_previous": self.preserve_previous,
            "similarity_warning": self.similarity_warning,
            "reason": self.reason,
        }


@dataclass
class ConsistencyMetrics:
    """Frame similarity assessment against reference profile."""
    similarity: float
    color_similarity: float
    edge_similarity: float
    warning: bool


@dataclass
class ConsistencyReport:
    """Comprehensive video quality and identity drift report."""
    frames: int
    fps: float
    duration_seconds: float
    mean_similarity: float
    minimum_similarity: float
    maximum_similarity: float
    warning_frame_count: int
    warning_ratio: float
    frame_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames": self.frames,
            "fps": round(self.fps, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "mean_similarity": round(self.mean_similarity, 4),
            "minimum_similarity": round(self.minimum_similarity, 4),
            "maximum_similarity": round(self.maximum_similarity, 4),
            "warning_frame_count": self.warning_frame_count,
            "warning_ratio": round(self.warning_ratio, 4),
            "frame_scores": [round(s, 4) for s in self.frame_scores],
        }
