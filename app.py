"""Interactive Production Studio Web Application for Animated Creator."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import cv2
import gradio as gr
import numpy as np

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.project import Project
from src.core.pipeline import PipelineManager
from src.core.jobs import JobManager
from src.core.recovery import recover_project
from src.core.hardware import system_info, select_live_backend
from src.art.opencv_renderer import OpenCVArtRenderer
from export_youtube import export, PRESETS


def load_style_choices():
    styles_path = Path("styles.json")
    if styles_path.exists():
        with open(styles_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data.keys())
    return ["anime_creator", "clean_illustration", "comic", "watercolor", "manga"]


job_manager = JobManager()
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


def process_video_pipeline(input_video, style_choice):
    if not input_video:
        return None, "Please upload or select an input video."

    in_path = Path(input_video)
    project_dir = Path("projects") / in_path.stem
    project = Project(project_dir)

    if not project.manifest_path.exists():
        project.create(in_path.stem)

    recover_project(project)
    pipeline = PipelineManager(project)

    try:
        result = pipeline.run(
            input_video=in_path,
            style=style_choice or "anime_creator",
        )
        master_path = result["final_video"]
        summary_msg = f"✅ Processing Complete!\n\nProject: {project.root.resolve()}\nMaster: {master_path}\nValidation: {result.get('validation')}"
        return master_path, summary_msg
    except Exception as exc:
        return None, f"❌ Pipeline failed: {str(exc)}"


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
    return system_info()


def create_app():
    style_choices = load_style_choices()

    with gr.Blocks(title="Animated Creator Studio", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎬 ANIMATED CREATOR STUDIO (Phase 6 Production)
            ### Real Video to Consistent Anime/Illustrated YouTube Content Pipeline
            """
        )

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
                        run_btn = gr.Button("⚡ Render Full Animated Creator Pipeline", variant="primary")
                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Final YouTube Master Output")
                        status_output = gr.Textbox(label="Status & Execution Telemetry", lines=6)

                run_btn.click(
                    fn=process_video_pipeline,
                    inputs=[video_input, style_dropdown],
                    outputs=[video_output, status_output],
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
                diag_json = gr.JSON(label="Hardware Report", value=get_system_diagnostics)
                diag_btn.click(fn=get_system_diagnostics, outputs=[diag_json])

    return demo


def main():
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
