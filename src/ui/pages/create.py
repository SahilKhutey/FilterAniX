from __future__ import annotations

import gradio as gr
from src.ui.callbacks import load_style_choices
from src.ui.components.parameter_panel import create_mathematical_parameter_panel


def render_create_page():
    """Renders the primary Project Creation & Setup workflow."""
    gr.Markdown("### 🎬 Create Video Project")

    with gr.Row():
        project_name_input = gr.Textbox(
            label="Project Name",
            placeholder="e.g. My Anime Creator Video",
            value="my_anime_video",
        )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input Creator Video (Drag & Drop or Browse)")
            inspect_btn = gr.Button("🔍 Inspect & Load Video Metadata", variant="secondary")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Input Metadata Detected")
                meta_res = gr.Textbox(label="Resolution", value="-", interactive=False)
                meta_fps = gr.Textbox(label="Framerate", value="-", interactive=False)
                meta_frames = gr.Textbox(label="Total Frames", value="-", interactive=False)
                meta_dur = gr.Textbox(label="Duration", value="-", interactive=False)
                meta_audio = gr.Textbox(label="Audio Track", value="-", interactive=False)

    with gr.Row():
        with gr.Column(scale=1):
            style_choices = load_style_choices()
            style_dropdown = gr.Dropdown(
                choices=style_choices,
                value=style_choices[0] if style_choices else "anime_creator",
                label="Artistic Preset",
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                **Deterministic Mathematical Rendering**:
                Uses MTH-02 through MTH-10 deterministic fields. No random diffusion seed jitter, 
                rock-solid temporal stability, and zero GPU memory spikes.
                """
            )

    # Full Mathematical Parameters Panel
    math_controls = create_mathematical_parameter_panel()

    with gr.Row():
        create_start_btn = gr.Button("⚡ Start Production Render Job", variant="primary", size="lg")

    create_status_msg = gr.Textbox(label="Execution Status", lines=2, interactive=False)

    return {
        "project_name": project_name_input,
        "video_input": video_input,
        "inspect_btn": inspect_btn,
        "meta_res": meta_res,
        "meta_fps": meta_fps,
        "meta_frames": meta_frames,
        "meta_dur": meta_dur,
        "meta_audio": meta_audio,
        "style_dropdown": style_dropdown,
        "math_controls": math_controls,
        "start_btn": create_start_btn,
        "status_msg": create_status_msg,
    }
