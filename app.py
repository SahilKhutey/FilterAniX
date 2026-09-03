"""Interactive Production Studio Web Application for Animated Creator (Phase 6 / P3 Production Hardening)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.project import Project
from src.core.pipeline import PipelineManager
from src.core.jobs import JobManager, Job
from src.core.recovery import recover_project
from src.core.resource_monitor import ResourceMonitor
from src.core.hardware import system_info
from src.art.opencv_renderer import OpenCVArtRenderer
from export_youtube import export, PRESETS


def load_style_choices():
    styles_path = Path("styles.json")
    if styles_path.exists():
        with open(styles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data.keys())
    return ["anime_creator", "clean_illustration", "comic", "watercolor", "manga"]


job_manager = JobManager(max_workers=1)
_live_renderer = OpenCVArtRenderer()


def live_camera_stylize(frame):
    """Processes webcam frames in real-time with lightweight OpenCV stylization."""
    if frame is None:
        return None

    h, w = frame.shape[:2]
    if w > 640:
        small = cv2.resize(frame, (640, int(h * 640 / w)))
        art = _live_renderer.render(small)
        return cv2.resize(art, (w, h))

    return _live_renderer.render(frame)


def run_pipeline_job(
    project: Project,
    input_path: Path,
    style: str,
    job: Optional[Job] = None,
):
    pipeline = PipelineManager(project)
    return pipeline.run(
        input_video=input_path,
        style=style,
        job=job,
    )


def start_video_job(
    input_video: Optional[str],
    style_choice: Optional[str],
) -> Tuple[str, str]:
    if not input_video:
        return "", "❌ Please upload or select an input creator video."

    in_path = Path(input_video)
    project_dir = Path("projects") / in_path.stem
    project = Project(project_dir)

    if not project.manifest_path.exists():
        project.create(in_path.stem)

    recover_project(project)
    job = job_manager.create()

    job_manager.run_async(
        job,
        run_pipeline_job,
        project,
        in_path,
        style_choice or "anime_creator",
    )

    return (
        job.job_id,
        f"⏳ Job Queued: {job.job_id}\nProject: {project.root.resolve()}",
    )


def get_job_status(job_id: str) -> Tuple[str, Optional[str]]:
    if not job_id:
        return "No active job selected.", None

    status = job_manager.status(job_id)
    if status.get("status") == "not_found":
        return f"Job [{job_id}] not found.", None

    state = status.get("status", "unknown").upper()
    stage = status.get("stage", "queued")
    progress = status.get("progress", 0.0) * 100.0
    current_f = status.get("current_frame", 0)
    total_f = status.get("total_frames", 0)
    fps = status.get("fps", 0.0)
    eta = status.get("eta_seconds", 0.0)
    msg = status.get("message", "")

    message = (
        f"═══════════════════════════════════════\n"
        f" Job ID:   {job_id}\n"
        f" Status:   {state}\n"
        f" Stage:    {stage}\n"
        f" Progress: {progress:5.1f}%\n"
        f" Frames:   {current_f}/{total_f} (FPS: {fps:.2f}, ETA: {eta:.1f}s)\n"
        f" Detail:   {msg}\n"
        f"═══════════════════════════════════════"
    )

    result_video = None
    if status.get("status") == "complete":
        res = status.get("result")
        if isinstance(res, dict):
            result_video = res.get("final_video")
            message += f"\n\n✅ Production Master Ready: {result_video}"
    elif status.get("status") == "failed":
        err = status.get("error", "Unknown error")
        message += f"\n\n❌ Job Failed: {err}"
    elif status.get("status") == "cancelled":
        message += f"\n\n⛔ Job was cancelled by user."

    return message, result_video


def pause_job(job_id: str) -> str:
    if not job_id:
        return "No active job to pause."
    if job_manager.pause(job_id):
        return f"⏸ Job [{job_id}] paused at safe boundary."
    return f"Unable to pause job [{job_id}]."


def resume_job(job_id: str) -> str:
    if not job_id:
        return "No active job to resume."
    if job_manager.resume(job_id):
        return f"▶ Job [{job_id}] resumed."
    return f"Unable to resume job [{job_id}]."


def cancel_job(job_id: str) -> str:
    if not job_id:
        return "No active job to cancel."
    if job_manager.cancel(job_id):
        return f"⛔ Cancellation requested for job [{job_id}]."
    return f"Unable to cancel job [{job_id}]."


def export_preset_video(master_video, preset_choice):
    if not master_video:
        return None, "Please provide a master video to export."

    in_p = Path(master_video)
    out_p = in_p.parent / f"youtube_{preset_choice}.mp4"

    try:
        exported = export(
            input_path=in_p,
            output_path=out_p,
            preset=preset_choice,
        )
        return str(exported), f"✅ Exported YouTube {preset_choice} Master to:\n{exported}"
    except Exception as exc:
        return None, f"❌ Export failed: {str(exc)}"


def get_system_diagnostics():
    snap = ResourceMonitor.snapshot(".")
    info = system_info()
    return {
        "hardware_info": info,
        "runtime_resources": snap,
    }


def create_app():
    style_choices = load_style_choices()

    with gr.Blocks(title="FilterAniX Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 FilterAniX — Production Creator Studio (P3 Infrastructure)
            ### Background Job Queue • Identity Consistency • Dynamic Control • Broadcast Output
            """
        )

        job_id_state = gr.State("")

        with gr.Tabs():
            # TAB 1: Main Creator Video Pipeline
            with gr.TabItem("🎬 Process Creator Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Input Creator Video")
                        style_dropdown = gr.Dropdown(
                            choices=style_choices,
                            value=style_choices[0] if style_choices else "anime_creator",
                            label="Artistic Style Preset",
                        )
                        start_btn = gr.Button("⚡ Start Background Render Job", variant="primary")

                        with gr.Row():
                            pause_btn = gr.Button("⏸ Pause", variant="secondary")
                            resume_btn = gr.Button("▶ Resume", variant="secondary")
                            cancel_btn = gr.Button("⛔ Cancel", variant="stop")

                    with gr.Column(scale=1):
                        status_output = gr.Textbox(
                            label="Job Status & Execution Telemetry",
                            lines=10,
                            value="No job running.",
                        )
                        video_output = gr.Video(label="Final YouTube Master Output")

                # Wire start button
                start_btn.click(
                    fn=start_video_job,
                    inputs=[video_input, style_dropdown],
                    outputs=[job_id_state, status_output],
                )

                # Wire control buttons
                pause_btn.click(
                    fn=pause_job,
                    inputs=[job_id_state],
                    outputs=[status_output],
                )
                resume_btn.click(
                    fn=resume_job,
                    inputs=[job_id_state],
                    outputs=[status_output],
                )
                cancel_btn.click(
                    fn=cancel_job,
                    inputs=[job_id_state],
                    outputs=[status_output],
                )

                # Automatic UI Polling via gr.Timer
                status_timer = gr.Timer(value=1.0, active=True)
                status_timer.tick(
                    fn=get_job_status,
                    inputs=[job_id_state],
                    outputs=[status_output, video_output],
                )

            # TAB 2: Live Camera Mode
            with gr.TabItem("📹 Live Camera"):
                gr.Markdown("### Real-Time Live Stylizer Preview (Fast Mode)")
                with gr.Row():
                    cam_input = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam Input")
                    cam_output = gr.Image(label="Live Animated Creator Preview")

                cam_input.stream(fn=live_camera_stylize, inputs=[cam_input], outputs=[cam_output])

            # TAB 3: YouTube Multi-Resolution Export
            with gr.TabItem("📤 YouTube Export"):
                with gr.Row():
                    with gr.Column():
                        export_video_in = gr.Video(label="Input Master Video")
                        preset_dropdown = gr.Dropdown(
                            choices=list(PRESETS.keys()),
                            value="1080p",
                            label="YouTube Preset Resolution",
                        )
                        export_btn = gr.Button("🚀 Export YouTube Optimized MP4", variant="primary")
                    with gr.Column():
                        exported_video_out = gr.Video(label="Exported MP4")
                        export_status = gr.Textbox(label="Export Status", lines=3)

                export_btn.click(
                    fn=export_preset_video,
                    inputs=[export_video_in, preset_dropdown],
                    outputs=[exported_video_out, export_status],
                )

            # TAB 4: System Diagnostics
            with gr.TabItem("💻 System Diagnostics"):
                diag_btn = gr.Button("🔄 Refresh Hardware & Capability Report")
                diag_json = gr.JSON(label="Hardware & Resource Report", value=get_system_diagnostics)
                diag_btn.click(fn=get_system_diagnostics, outputs=[diag_json])

    return demo


def main():
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
