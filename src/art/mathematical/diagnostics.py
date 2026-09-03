"""Diagnostic and Performance Telemetry for Mathematical Anime Engine."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FrameDiagnostic:
    frame_index: int
    duration_ms: float
    motion_score: float
    edge_density: float
    mean_luminance: float


@dataclass
class EngineBenchmarkSummary:
    total_frames: int = 0
    total_duration_sec: float = 0.0
    average_fps: float = 0.0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0


class MathematicalEngineDiagnostics:
    """Tracks per-frame latency, field metrics, and runtime throughput."""

    def __init__(self):
        self.frame_records: List[FrameDiagnostic] = []
        self._start_time: Optional[float] = None

    def start_run(self) -> None:
        self.frame_records.clear()
        self._start_time = time.perf_counter()

    def record_frame(
        self,
        frame_index: int,
        duration_ms: float,
        motion_score: float = 0.0,
        edge_density: float = 0.0,
        mean_luminance: float = 0.0,
    ) -> None:
        self.frame_records.append(
            FrameDiagnostic(
                frame_index=frame_index,
                duration_ms=duration_ms,
                motion_score=motion_score,
                edge_density=edge_density,
                mean_luminance=mean_luminance,
            )
        )

    def summarize(self) -> EngineBenchmarkSummary:
        if not self.frame_records:
            return EngineBenchmarkSummary()

        total_frames = len(self.frame_records)
        latencies = [f.duration_ms for f in self.frame_records]
        total_time = sum(latencies) / 1000.0
        avg_latency = float(sum(latencies) / total_frames)
        sorted_latencies = sorted(latencies)
        p95_idx = min(total_frames - 1, int(0.95 * total_frames))
        p95_latency = float(sorted_latencies[p95_idx])

        fps = float(total_frames / total_time) if total_time > 0 else 0.0

        return EngineBenchmarkSummary(
            total_frames=total_frames,
            total_duration_sec=round(total_time, 3),
            average_fps=round(fps, 2),
            average_latency_ms=round(avg_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
        )
