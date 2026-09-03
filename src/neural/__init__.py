"""FilterAniX Neural Assistance Package (Enable / Disable Feature - Not Core).

Pluggable lightweight neural assistive layer for scene understanding:
  - Salient object segmentation (U²-Netp)
  - Portrait alpha matting (MODNet)
  - Dense optical flow (Micron-Flow)
  - Relative scene depth (Depth Anything V2 Small INT8)
"""
from __future__ import annotations

from .config import NeuralAssistConfig
from .registry import (
    ModelRegistry,
    ModelSpec,
    MODEL_SPECS,
    MAX_TOTAL_MODEL_BUDGET_MB,
    RECOMMENDED_DEFAULT_BUNDLE_MB,
)
from .runtime import ONNXRuntimeManager
from .memory import NeuralMemoryTracker, NeuralTelemetrySnapshot
from .segmentation.u2netp import U2NetpRunner
from .segmentation.modnet import MODNetRunner
from .motion.micron_flow import MicronFlowRunner
from .depth.depth_anything import DepthAnythingRunner
from .manager import NeuralAssistManager

__all__ = [
    "NeuralAssistConfig",
    "ModelRegistry",
    "ModelSpec",
    "MODEL_SPECS",
    "MAX_TOTAL_MODEL_BUDGET_MB",
    "RECOMMENDED_DEFAULT_BUNDLE_MB",
    "ONNXRuntimeManager",
    "NeuralMemoryTracker",
    "NeuralTelemetrySnapshot",
    "U2NetpRunner",
    "MODNetRunner",
    "MicronFlowRunner",
    "DepthAnythingRunner",
    "NeuralAssistManager",
]
