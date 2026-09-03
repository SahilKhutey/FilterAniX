"""FilterAniX Neural Memory & Latency Tracker.

Monitors:
  - Model disk footprint vs budget (<= 1024 MB)
  - Runtime execution latency (ms) per model
  - Active neural execution telemetry
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from .registry import MAX_TOTAL_MODEL_BUDGET_MB, ModelRegistry


@dataclass
class NeuralTelemetrySnapshot:
    """Live telemetry record of neural assist models."""
    enabled: bool
    models_on_disk_mb: float
    budget_limit_mb: float
    active_models: list[str]
    latencies_ms: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "models_on_disk_mb": round(self.models_on_disk_mb, 2),
            "budget_limit_mb": self.budget_limit_mb,
            "budget_used_ratio": round(self.models_on_disk_mb / max(1.0, self.budget_limit_mb), 3),
            "active_models": list(self.active_models),
            "latencies_ms": {k: round(v, 2) for k, v in self.latencies_ms.items()},
        }


class NeuralMemoryTracker:
    """Tracks neural model storage and runtime performance."""

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self._latencies: Dict[str, float] = {}
        self._active_models: set[str] = set()

    def record_latency(self, model_key: str, latency_ms: float) -> None:
        self._latencies[model_key] = float(latency_ms)
        self._active_models.add(model_key)

    def get_disk_usage_mb(self) -> float:
        """Calculates total size of all models physically stored in models_dir."""
        if not self.models_dir.exists():
            return 0.0
        total_bytes = 0
        for f in self.models_dir.glob("*.*"):
            if f.is_file():
                total_bytes += f.stat().st_size
        return total_bytes / (1024.0 * 1024.0)

    def snapshot(self, enabled: bool = False) -> NeuralTelemetrySnapshot:
        return NeuralTelemetrySnapshot(
            enabled=enabled,
            models_on_disk_mb=self.get_disk_usage_mb(),
            budget_limit_mb=MAX_TOTAL_MODEL_BUDGET_MB,
            active_models=sorted(list(self._active_models)),
            latencies_ms=dict(self._latencies),
        )
