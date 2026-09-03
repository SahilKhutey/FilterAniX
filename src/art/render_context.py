from dataclasses import dataclass
from typing import Optional


@dataclass
class RenderContext:
    """Encapsulates frame-level control signals from Vision, Temporal Planner, and Lip-Sync."""
    frame_index: int
    scene_id: int = 0
    scene_cut: bool = False
    is_keyframe: bool = False
    preserve_previous: bool = False
    reference_strength: float = 0.55
    viseme: str = "closed"
