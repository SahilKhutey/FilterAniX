"""FilterAniX ONNX Runtime Session Manager.

Safely handles execution providers (CPU, DirectML, CUDA) with zero-crash fallback
when onnxruntime is absent or weights are unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger("filteranix.neural.runtime")

_ORT_AVAILABLE: bool = False
_ORT_MODULE: Optional[Any] = None

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
    _ORT_MODULE = ort
except ImportError:
    _ORT_AVAILABLE = False
    _ORT_MODULE = None


class ONNXRuntimeManager:
    """Manages ONNX Runtime sessions, provider selection, and graceful fallback."""

    @staticmethod
    def is_available() -> bool:
        """Returns True if onnxruntime is installed and importable."""
        return _ORT_AVAILABLE

    @staticmethod
    def get_available_providers() -> List[str]:
        if not _ORT_AVAILABLE or _ORT_MODULE is None:
            return []
        try:
            return list(_ORT_MODULE.get_available_providers())
        except Exception:
            return ["CPUExecutionProvider"]

    @classmethod
    def select_providers(cls, preference: str = "auto") -> List[str]:
        """Chooses execution providers according to hardware capabilities and preference."""
        available = cls.get_available_providers()
        if not available:
            return ["CPUExecutionProvider"]

        pref = (preference or "auto").lower()
        if pref == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if pref in ("directml", "dml") and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if pref == "cpu":
            return ["CPUExecutionProvider"]

        # Auto: Prefer DirectML/CUDA if present, otherwise CPU
        providers = []
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
        elif "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @classmethod
    def create_session(
        cls,
        model_path: str | Path,
        device_preference: str = "auto",
    ) -> Optional[Any]:
        """
        Instantiates an InferenceSession for the given ONNX model.
        Returns None if onnxruntime is missing or loading fails, allowing callers to fall back.
        """
        if not _ORT_AVAILABLE or _ORT_MODULE is None:
            logger.debug("onnxruntime is not installed; neural runner will use classical fallback.")
            return None

        p = Path(model_path)
        if not p.exists():
            logger.debug(f"Model file not found at {p}; neural runner will use classical fallback.")
            return None

        try:
            opts = _ORT_MODULE.SessionOptions()
            opts.graph_optimization_level = _ORT_MODULE.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4

            providers = cls.select_providers(device_preference)
            session = _ORT_MODULE.InferenceSession(str(p), sess_options=opts, providers=providers)
            return session
        except Exception as exc:
            logger.warning(f"Failed to load ONNX model {p}: {exc}. Using classical fallback.")
            return None
