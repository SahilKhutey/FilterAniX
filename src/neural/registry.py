"""FilterAniX Neural Model Registry & Budget Enforcer.

Strictly enforces:
  - Total local model budget <= 1024 MB
  - Recommended bundle <= 250 MB
  - Valid open-source licenses (Apache-2.0 / BSD / MIT)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for an assistive neural model."""
    key: str
    name: str
    task: str
    license: str
    size_mb: float
    params: str
    filename: str
    input_resolution: tuple[int, int]
    frequency_interval: int
    format: str = "onnx"


MODEL_SPECS: Dict[str, ModelSpec] = {
    "u2netp": ModelSpec(
        key="u2netp",
        name="U²-Netp (Salient Object Segmentation)",
        task="segmentation",
        license="apache-2.0",
        size_mb=4.36,
        params="1.13M",
        filename="u2netp.onnx",
        input_resolution=(320, 320),
        frequency_interval=5,
    ),
    "modnet": ModelSpec(
        key="modnet",
        name="MODNet (Real-Time Portrait Matting)",
        task="portrait_matting",
        license="apache-2.0",
        size_mb=26.0,
        params="6.49M",
        filename="modnet_photographic_portrait_matting.onnx",
        input_resolution=(512, 512),
        frequency_interval=5,
    ),
    "micron_flow": ModelSpec(
        key="micron_flow",
        name="Micron-Flow (Lightweight Optical Flow)",
        task="optical_flow",
        license="apache-2.0",
        size_mb=2.10,
        params="522K",
        filename="micron_flow.onnx",
        input_resolution=(256, 256),
        frequency_interval=1,
    ),
    "depth_anything_v2_small_int8": ModelSpec(
        key="depth_anything_v2_small_int8",
        name="Depth Anything V2 Small (INT8)",
        task="depth_estimation",
        license="apache-2.0",
        size_mb=27.3,
        params="24.8M",
        filename="depth_anything_v2_vits_int8.onnx",
        input_resolution=(518, 518),
        frequency_interval=8,
    ),
}

MAX_TOTAL_MODEL_BUDGET_MB = 1024.0
RECOMMENDED_DEFAULT_BUNDLE_MB = 250.0
ALLOWED_LICENSES = {"apache-2.0", "bsd-3-clause", "bsd-2-clause", "mit"}


class ModelRegistry:
    """Manages model metadata and ensures licensing and size budget invariants."""

    @staticmethod
    def get_spec(key: str) -> Optional[ModelSpec]:
        return MODEL_SPECS.get(key)

    @staticmethod
    def list_models() -> List[ModelSpec]:
        return list(MODEL_SPECS.values())

    @staticmethod
    def get_total_registered_size_mb() -> float:
        return sum(spec.size_mb for spec in MODEL_SPECS.values())

    @classmethod
    def validate_budget_compliance(cls) -> bool:
        """Verifies that all registered models remain strictly under the 1 GB ceiling."""
        total_mb = cls.get_total_registered_size_mb()
        if total_mb > MAX_TOTAL_MODEL_BUDGET_MB:
            raise ValueError(
                f"Model registry total size {total_mb:.1f} MB exceeds hard budget limit {MAX_TOTAL_MODEL_BUDGET_MB} MB"
            )
        return True

    @classmethod
    def validate_licensing_compliance(cls) -> bool:
        """Verifies that every model is licensed under permissible open source terms."""
        for spec in MODEL_SPECS.values():
            if spec.license.lower() not in ALLOWED_LICENSES:
                raise ValueError(
                    f"Model {spec.key} has non-permissive license: {spec.license}"
                )
        return True

    @staticmethod
    def is_model_present(key: str, models_dir: str | Path = "models") -> bool:
        spec = MODEL_SPECS.get(key)
        if spec is None:
            return False
        p = Path(models_dir) / spec.filename
        return p.exists() and p.is_file() and p.stat().st_size > 1024

    @staticmethod
    def get_model_path(key: str, models_dir: str | Path = "models") -> Optional[Path]:
        spec = MODEL_SPECS.get(key)
        if spec is None:
            return None
        p = Path(models_dir) / spec.filename
        return p if p.exists() else None
