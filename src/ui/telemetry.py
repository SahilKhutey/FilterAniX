from __future__ import annotations

import os
from typing import Any, Dict, Optional
import psutil

from src.core.jobs import JobManager
from src.ui.state import UIState, ui_state_manager

# Pipeline stage display order
PIPELINE_STAGES = [
    ("input", "INPUT"),
    ("vision", "VISION"),
    ("consistency", "TEMPORAL / IDENTITY PLANNING"),
    ("artistic", "MATHEMATICAL ART RENDERING"),
    ("lipsync", "LIP-SYNC & VISEME TIMELINE"),
    ("composition", "MEDIA COMPOSITION"),
    ("validation", "BROADCAST VALIDATION"),
    ("export", "MASTER EXPORT"),
]

MTH_SUBSTAGES = [
    ("mth02", "MTH-02 Color Field"),
    ("mth03", "MTH-03 Tone Field"),
    ("mth04", "MTH-04 Palette Field"),
    ("mth05", "MTH-05 Edge Field"),
    ("mth06", "MTH-06 Shadow/Highlight"),
    ("mth07", "MTH-07 Geometry"),
    ("mth08", "MTH-08 Face"),
    ("mth09", "MTH-09 Lighting"),
    ("mth10", "MTH-10 Temporal"),
]


def get_current_memory_gb() -> float:
    """Returns process RSS memory in GB."""
    try:
        proc = psutil.Process(os.getpid())
        return round(proc.memory_info().rss / (1024 ** 3), 2)
    except Exception:
        return 0.0


def get_job_state(job_manager: JobManager, job_id: str) -> Dict[str, Any]:
    """Retrieves snapshot for active job, or default idle state."""
    if not job_id:
        return {
            "job_id": "",
            "id": "",
            "status": "idle",
            "stage": "",
            "substage": "",
            "progress": 0.0,
            "frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "eta": 0.0,
            "eta_seconds": 0.0,
            "message": "System ready. No active job.",
            "current_output": None,
            "error": None,
            "result": None,
        }

    snapshot = job_manager.snapshot(job_id)
    if snapshot is None:
        return {
            "job_id": job_id,
            "id": job_id,
            "status": "unknown",
            "stage": "",
            "substage": "",
            "progress": 0.0,
            "frame": 0,
            "total_frames": 0,
            "fps": 0.0,
            "eta": 0.0,
            "eta_seconds": 0.0,
            "message": f"Job [{job_id}] not found.",
            "current_output": None,
            "error": "Job not found",
            "result": None,
        }

    return snapshot


def format_status_bar(snapshot: Dict[str, Any]) -> str:
    """Formats bottom telemetry string."""
    status = snapshot.get("status", "idle").upper()
    job_id = snapshot.get("job_id", "")
    stage = snapshot.get("stage", "")
    substage = snapshot.get("substage", "")
    progress = snapshot.get("progress", 0.0) * 100.0
    frame = snapshot.get("frame", 0)
    total_frames = snapshot.get("total_frames", 0)
    fps = snapshot.get("fps", 0.0)
    eta = snapshot.get("eta_seconds", 0.0)
    mem = get_current_memory_gb()

    if status == "IDLE":
        return f"● SYSTEM READY   |   Memory: {mem:.1f} GB   |   No active render"

    active_stage_str = f"{stage}"
    if substage:
        active_stage_str += f" ({substage})"

    eta_str = f"{int(eta // 60):02d}:{int(eta % 60):02d}"

    return (
        f"Job: {job_id}  |  Status: {status}  |  Stage: {active_stage_str}  |  "
        f"{progress:4.1f}%  |  Frame: {frame}/{total_frames}  |  "
        f"FPS: {fps:.1f}  |  ETA: {eta_str}  |  RAM: {mem:.1f} GB"
    )


def format_pipeline_stepper(snapshot: Dict[str, Any]) -> str:
    """
    Renders clean text representation of the 8 production pipeline stages
    and MTH-02 through MTH-10 sub-stages with live status markers.
    """
    current_stage = snapshot.get("stage", "").lower()
    current_status = snapshot.get("status", "idle").lower()
    progress = snapshot.get("progress", 0.0)

    lines = ["PIPELINE EXECUTION MONITOR", "═" * 46]

    stage_order = [s[0] for s in PIPELINE_STAGES]
    curr_idx = -1
    if current_stage in stage_order:
        curr_idx = stage_order.index(current_stage)
    elif current_status == "complete":
        curr_idx = len(stage_order)

    for idx, (key, label) in enumerate(PIPELINE_STAGES):
        if idx < curr_idx or current_status == "complete":
            marker = "✓"
            status_text = "COMPLETE"
        elif idx == curr_idx:
            if current_status == "failed":
                marker = "⚠"
                status_text = "FAILED"
            elif current_status == "paused":
                marker = "⏸"
                status_text = "PAUSED"
            else:
                marker = "●"
                status_text = f"{progress * 100:.1f}%"
        else:
            marker = "○"
            status_text = "PENDING"

        lines.append(f" {marker}  {label:<32} {status_text}")

        # If artistic stage is currently active or complete, show MTH sub-stages!
        if key == "artistic" and (idx == curr_idx or idx < curr_idx or current_status == "complete"):
            frame = snapshot.get("frame", 0)
            total_f = snapshot.get("total_frames", 1)
            f_pct = frame / max(1, total_f)

            for sub_idx, (mth_key, mth_label) in enumerate(MTH_SUBSTAGES):
                sub_threshold = (sub_idx + 1) / len(MTH_SUBSTAGES)
                if idx < curr_idx or current_status == "complete" or f_pct >= sub_threshold:
                    s_mark = "✓"
                elif f_pct >= (sub_idx / len(MTH_SUBSTAGES)):
                    s_mark = "●"
                else:
                    s_mark = "○"
                prefix = " └─" if sub_idx == len(MTH_SUBSTAGES) - 1 else " ├─"
                lines.append(f"    {prefix} {mth_label:<28} {s_mark}")

    lines.append("═" * 46)
    return "\n".join(lines)
