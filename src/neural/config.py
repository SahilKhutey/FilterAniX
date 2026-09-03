"""FilterAniX Neural Assistance Configuration.

Neural models are strictly assistive and pluggable (Enable / Disable feature).
They assist Phase 2 Vision in understanding the scene without replacing the mathematical engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class NeuralAssistConfig:
    """Configuration for the pluggable Neural Assistance layer."""
    enabled: bool = False

    # Task Toggles
    use_segmentation: bool = True     # U²-Netp (Salient subject segmentation)
    use_matting: bool = True          # MODNet (Portrait alpha matting for Creator mode)
    use_flow: bool = True             # Micron-Flow (Lightweight neural optical flow)
    use_depth: bool = False           # Depth Anything V2 Small INT8 (Cinematic depth)

    # Multi-Rate Frequency Intervals (Run on keyframes and temporally propagate)
    segmentation_interval: int = 5    # Every 5 frames
    matting_interval: int = 5         # Every 5 frames
    depth_interval: int = 8           # Every 8 frames
    flow_interval: int = 1            # Every frame

    # Hardware & Runtime Execution
    device_preference: str = "auto"   # "auto", "cpu", "cuda", "directml"
    models_dir: str = "models"
    max_model_budget_mb: float = 1024.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "use_segmentation": self.use_segmentation,
            "use_matting": self.use_matting,
            "use_flow": self.use_flow,
            "use_depth": self.use_depth,
            "segmentation_interval": self.segmentation_interval,
            "matting_interval": self.matting_interval,
            "depth_interval": self.depth_interval,
            "flow_interval": self.flow_interval,
            "device_preference": self.device_preference,
            "models_dir": self.models_dir,
            "max_model_budget_mb": self.max_model_budget_mb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NeuralAssistConfig:
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)
