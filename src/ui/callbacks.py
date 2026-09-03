from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.art.opencv_renderer import FastPreviewRenderer, OpenCVArtRenderer
from src.art.mathematical import (
    MathematicalAnimeEngine,
    MathematicalAnimeStyle,
    MathematicalRenderer,
    validate_render,
)
from src.vision.face_engine import FaceEngine
from src.vision.models import FrameVisionData
from src.core.jobs import Job, JobManager
from src.core.pipeline import PipelineManager
from src.core.project import Project
from src.core.recovery import recover_project
from src.core.resource_monitor import ResourceMonitor
from src.core.hardware import system_info
from src.io.video_io import inspect_video
from src.ui.state import UIState, ui_state_manager
from export_youtube import export, PRESETS
from src.neural import NeuralAssistConfig, NeuralAssistManager


# First-class Live Mathematical Anime Engine and Vision Tracking
_live_math_engine = MathematicalAnimeEngine(MathematicalAnimeStyle.creator_anime())
_fast_preview_renderer = FastPreviewRenderer()
_live_face_engine = FaceEngine()
_live_frame_counter = 0

# Pluggable Neural Assistance Manager (Enable/Disable Feature - Not Core)
_neural_config = NeuralAssistConfig(enabled=False)
_neural_manager = NeuralAssistManager(_neural_config)

# Style Lab intermediate stages cache
_last_style_lab_stages: Dict[str, np.ndarray] = {}
_last_style_lab_diagnostics: str = "Awaiting frame render."


def load_style_choices() -> List[str]:
    """Loads style names from styles.json or default built-in anime styles."""
    styles_path = Path("styles.json")
    if styles_path.exists():
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return list(data.keys())
        except Exception:
            pass
    return ["anime_creator", "clean_illustration", "comic", "watercolor", "manga"]


def inspect_input_video_file(video_path: Optional[str]) -> Tuple[str, str, str, str, str]:
    """Inspects uploaded video file and returns metadata strings."""
    if not video_path:
        return "-", "-", "-", "-", "No"

    try:
        meta = inspect_video(video_path)
        resolution_str = f"{meta.width} × {meta.height}"
        fps_str = f"{meta.fps:.1f} FPS"
        frames_str = f"{meta.frame_count} frames"
        duration_str = f"{meta.duration_sec:.1f}s"
        audio_str = "✓ Present" if meta.has_audio else "None"

        ui_state_manager.set_state(UIState.INPUT_READY)
        ui_state_manager.add_notification(
            f"Video loaded: {resolution_str} @ {fps_str} ({duration_str}, {frames_str})",
            level="success",
        )
        return resolution_str, fps_str, frames_str, duration_str, audio_str
    except Exception as exc:
        ui_state_manager.add_notification(f"Failed to inspect video: {exc}", level="error")
        return "Error", "Error", "Error", "Error", "Error"


def run_pipeline_worker(
    project: Project,
    input_path: Path,
    style: str,
    style_params: Optional[dict] = None,
    job: Optional[Job] = None,
):
    """Executes the pipeline in the background worker thread."""
    pipeline = PipelineManager(project)
    return pipeline.run(
        input_video=input_path,
        style=style,
        job=job,
        style_params=style_params,
    )


def start_pipeline_job(
    job_manager: JobManager,
    input_video: Optional[str],
    project_name: Optional[str],
    style_choice: Optional[str],
    math_params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """Creates a project and non-blocking asynchronous render job."""
    if not input_video:
        return "", "❌ Please upload or select an input creator video.", ui_state_manager.get_button_states()

    in_path = Path(input_video).resolve()
    name = (project_name or in_path.stem).strip()
    if not name:
        name = "project_" + str(int(time.time()))

    project_dir = Path("projects") / name
    project = Project(project_dir)

    if not project.manifest_path.exists():
        project.create(name)

    recover_project(project)
    job = job_manager.create()
    ui_state_manager.set_active_job(job.job_id)
    ui_state_manager.set_active_project(name)
    ui_state_manager.set_state(UIState.RENDERING)
    ui_state_manager.add_notification(f"Render job [{job.job_id}] started for project '{name}'", level="info")

    job_manager.run_async(
        job,
        run_pipeline_worker,
        project,
        in_path,
        style_choice or "anime_creator",
        math_params,
    )

    btn_states = ui_state_manager.get_button_states(UIState.RENDERING)
    return (
        job.job_id,
        f"⏳ Job Queued: {job.job_id}\nProject: {project.root.resolve()}",
        btn_states,
    )


def pause_active_job(job_manager: JobManager, job_id: str) -> Tuple[str, Dict[str, Any]]:
    if not job_id:
        return "No active job to pause.", ui_state_manager.get_button_states()
    if job_manager.pause(job_id):
        ui_state_manager.set_state(UIState.PAUSED)
        ui_state_manager.add_notification(f"Job [{job_id}] paused at safe boundary.", level="warning")
        return f"⏸ Job [{job_id}] paused at safe boundary.", ui_state_manager.get_button_states(UIState.PAUSED)
    return f"Unable to pause job [{job_id}].", ui_state_manager.get_button_states()


def resume_active_job(job_manager: JobManager, job_id: str) -> Tuple[str, Dict[str, Any]]:
    if not job_id:
        return "No active job to resume.", ui_state_manager.get_button_states()
    if job_manager.resume(job_id):
        ui_state_manager.set_state(UIState.RENDERING)
        ui_state_manager.add_notification(f"Job [{job_id}] resumed.", level="info")
        return f"▶ Job [{job_id}] resumed.", ui_state_manager.get_button_states(UIState.RENDERING)
    return f"Unable to resume job [{job_id}].", ui_state_manager.get_button_states()


def stop_active_job(job_manager: JobManager, job_id: str) -> Tuple[str, Dict[str, Any]]:
    if not job_id:
        return "No active job to stop.", ui_state_manager.get_button_states()
    if job_manager.cancel(job_id):
        job = job_manager.get(job_id)
        if job is not None:
            job.status = "cancelled"
        ui_state_manager.set_state(UIState.IDLE)
        ui_state_manager.add_notification(f"Job [{job_id}] stopped by user.", level="warning")
        return f"⛔ Cancellation requested for job [{job_id}].", ui_state_manager.get_button_states(UIState.IDLE)
    return f"Unable to stop job [{job_id}].", ui_state_manager.get_button_states()


def retry_failed_job(job_manager: JobManager, job_id: str, input_video: Optional[str], style_choice: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    """Recovers the project and restarts rendering from the first failed stage."""
    project_name = ui_state_manager.active_project_name
    ui_state_manager.clear_error()
    return start_pipeline_job(job_manager, input_video, project_name, style_choice)


def export_preset_video(master_video: Optional[str], preset_choice: str) -> Tuple[Optional[str], str]:
    """Encodes master video into YouTube broadcast standard."""
    if not master_video:
        return None, "Please provide a master video to export."

    in_p = Path(master_video)
    if not in_p.exists():
        return None, f"Master video file not found: {master_video}"

    out_p = in_p.parent / f"youtube_{preset_choice}.mp4"

    try:
        exported = export(
            input_path=in_p,
            output_path=out_p,
            preset=preset_choice,
        )
        ui_state_manager.add_notification(f"Exported YouTube {preset_choice} Master to: {exported.name}", level="success")
        return str(exported), f"✅ Export completed:\n{exported}"
    except Exception as exc:
        ui_state_manager.add_notification(f"Export failed: {exc}", level="error")
        return None, f"❌ Export failed: {str(exc)}"


def render_style_lab_frame(
    video_path: Optional[str],
    frame_index: int,
    math_params: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Renders single frame through the MathematicalAnimeEngine for interactive Style Lab.
    Returns: (original_rgb, stylized_rgb, split_rgb, status_text)
    """
    global _last_style_lab_stages, _last_style_lab_diagnostics
    if not video_path:
        return None, None, None, "Please load a video first."

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, None, f"Could not open video: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = max(0, min(total_frames - 1, int(frame_index)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        return None, None, None, f"Could not read frame {frame_idx}"

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    enable_neural = bool(math_params.pop("enable_neural_assist", False))

    # Instantiate mathematical style
    try:
        custom_style = MathematicalAnimeStyle(**math_params).validated()
    except Exception:
        custom_style = MathematicalAnimeStyle.creator_anime()

    engine = MathematicalAnimeEngine(style=custom_style)
    # Detect face for vision guidance
    vision_obj = None
    try:
        faces = _live_face_engine.process(frame_rgb)
        vision_obj = FrameVisionData(
            frame_index=frame_idx,
            timestamp=0.0,
            width=frame_rgb.shape[1],
            height=frame_rgb.shape[0],
            faces=faces,
        )
    except Exception:
        pass

    # Optional Pluggable Neural Assistance Layer (MODNet / U²-Netp / Depth Anything V2)
    if enable_neural:
        _neural_config.enabled = True
        vision_obj = _neural_manager.process_frame(frame_rgb, vision_obj)
    else:
        _neural_config.enabled = False

    art_rgb, stages, telemetry = engine.render_stages(
        frame_rgb,
        vision_data=vision_obj,
        stabilize=False,
    )
    _last_style_lab_stages = stages
    _last_style_lab_diagnostics = engine.format_diagnostics_report(telemetry, frame_idx)

    # Generate split comparison (left: original, right: result)
    h, w = frame_rgb.shape[:2]
    split_x = w // 2
    split_rgb = frame_rgb.copy()
    split_rgb[:, split_x:] = art_rgb[:, split_x:]
    # Draw dividing line
    cv2.line(split_rgb, (split_x, 0), (split_x, h), (0, 255, 255), 2)

    status = (
        f"Rendered Frame {frame_idx + 1} / {total_frames} | "
        f"Edge: {custom_style.edge_strength:.2f} | Tone: {custom_style.tone_strength:.2f} | "
        f"Contrast: {custom_style.contrast:.2f} | Palette Mix: {custom_style.palette_mix:.2f}"
    )

    return frame_rgb, art_rgb, split_rgb, status


def get_style_lab_stage_frame(stage_name: str) -> Optional[np.ndarray]:
    """Returns the cached intermediate field image for the requested stage."""
    global _last_style_lab_stages
    return _last_style_lab_stages.get(stage_name, None)


def get_style_lab_diagnostics() -> str:
    """Returns the latest engine diagnostics text report."""
    global _last_style_lab_diagnostics
    return _last_style_lab_diagnostics


def save_style_preset(preset_name: str, math_params: Dict[str, Any]) -> str:
    """Saves custom mathematical parameter preset to styles.json."""
    if not preset_name or not preset_name.strip():
        return "❌ Please enter a valid preset name."

    key = preset_name.strip().lower().replace(" ", "_")
    styles_path = Path("styles.json")
    data = {}
    if styles_path.exists():
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}

    data[key] = {
        "name": preset_name.strip(),
        "description": f"Custom FilterAniX mathematical preset: {preset_name}",
        "renderer": "mathematical",
        **math_params,
    }

    try:
        with open(styles_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        ui_state_manager.add_notification(f"Saved style preset '{preset_name}'", level="success")
        return f"✅ Style preset '{preset_name}' saved successfully to styles.json!"
    except Exception as exc:
        return f"❌ Failed to save preset: {exc}"


def list_recent_projects() -> List[List[Any]]:
    """Scans projects/ directory and returns tabular list of recent projects."""
    projects_dir = Path("projects")
    if not projects_dir.exists():
        return []

    rows = []
    for p in sorted(projects_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_dir():
            manifest_file = p / "project.json"
            status = "unknown"
            created = "-"
            completed_stages = 0
            total_stages = 8
            if manifest_file.exists():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        m = json.load(f)
                        status = m.get("status", "created")
                        created = m.get("created_at", "-")[:19].replace("T", " ")
                        stages = m.get("stages", {})
                        completed_stages = sum(1 for s in stages.values() if s.get("status") == "complete")
                        total_stages = len(stages) or 8
                except Exception:
                    pass
            progress_str = f"{completed_stages}/{total_stages} stages"
            rows.append([p.name, status.upper(), progress_str, created])
    return rows


def live_camera_stylize_frame(
    frame: Optional[np.ndarray],
    mode: str = "🎨 MATHEMATICAL ENGINE",
    enable_neural: bool = False,
) -> Optional[np.ndarray]:
    """
    Real-time creator camera stylization.
    Supports:
      - 🎨 MATHEMATICAL ENGINE (Primary): Continuous image fields MTH-02..MTH-10 with Phase-2 Face tracking
      - ⚡ FAST PREVIEW (Fallback): Accelerated lightweight preview
      - Optional Neural Assistance: Pluggable MODNet portrait matting & Micron-Flow
    """
    global _live_frame_counter
    if frame is None:
        return None

    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return None

    h, w = rgb.shape[:2]
    _live_frame_counter += 1

    if "FAST" in str(mode).upper():
        # Fast preview mode (target max width 320 for 30 FPS)
        max_width = 320
        if w > max_width:
            new_h = max(2, int(h * max_width / w))
            small = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            small = rgb

        art = _fast_preview_renderer.render(small)
        if art.shape[:2] != (h, w):
            art = cv2.resize(art, (w, h), interpolation=cv2.INTER_LINEAR)
        return art

    # Canonical Mathematical Anime Engine path (target max width 640 for quality 15 FPS)
    max_width = 640
    if w > max_width:
        new_h = max(2, int(h * max_width / w))
        small = cv2.resize(rgb, (max_width, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = rgb

    # Phase 2 Vision analysis on live frame for Face / Geometry Field
    vision_data = None
    try:
        faces = _live_face_engine.process(small)
        vision_data = FrameVisionData(
            frame_index=_live_frame_counter,
            timestamp=time.time(),
            width=small.shape[1],
            height=small.shape[0],
            faces=faces,
        )
    except Exception:
        pass

    # Optional Pluggable Neural Assistance (MODNet / U²-Netp / Micron-Flow)
    if enable_neural:
        _neural_config.enabled = True
        vision_data = _neural_manager.process_frame(small, vision_data)
    else:
        _neural_config.enabled = False

    result = _live_math_engine.render(
        rgb=small,
        vision_data=vision_data,
        scene_cut=False,
        stabilize=True,
    )

    if result.shape[:2] != (h, w):
        result = cv2.resize(
            result,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    return result


def get_neural_telemetry_report() -> str:
    """Generates human-readable Neural Assistance budget and performance telemetry."""
    snapshot = _neural_manager.memory_tracker.snapshot(_neural_config.enabled)
    status_tag = "ACTIVE (Pluggable Assist)" if snapshot.enabled else "DISABLED (Classical Fallback)"
    active_models = ", ".join(snapshot.active_models) if snapshot.active_models else "None"
    lines = [
        f"Neural Assist Status: {status_tag}",
        f"Storage Footprint:   {snapshot.models_on_disk_mb:.1f} MB / {snapshot.budget_limit_mb:.0f} MB (Budget Limit)",
        f"Active Assist Models: {active_models}",
    ]
    if snapshot.latencies_ms:
        for m, lat in snapshot.latencies_ms.items():
            lines.append(f"  • {m}: {lat:.1f} ms")
    return "\n".join(lines)


def get_system_diagnostics_report() -> Dict[str, Any]:
    """Generates complete hardware and resource diagnostics report."""
    return {
        "hardware_info": system_info(),
        "runtime_resources": ResourceMonitor.snapshot("."),
        "neural_telemetry": _neural_manager.memory_tracker.snapshot(_neural_config.enabled).to_dict(),
    }
