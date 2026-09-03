from __future__ import annotations

import gradio as gr
from src.ui.components.preview import create_video_preview
from src.ui.components.pipeline_graph import create_pipeline_graph
from src.ui.components.job_monitor import create_control_buttons


def render_dashboard_page():
    """Renders the primary Studio Dashboard."""
    gr.Markdown("### 📊 Studio Operations Dashboard")

    # Status Cards
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**INPUT VIDEO**")
                card_res = gr.Textbox(value="No input loaded", label="Resolution & Duration", interactive=False)
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**VISION & KEYFRAMES**")
                card_vision = gr.Textbox(value="Awaiting Analysis", label="Vision Engine Status", interactive=False)
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("**ARTISTIC STYLE**")
                card_style = gr.Textbox(value="Anime Creator (Deterministic MTH)", label="Active Style Engine", interactive=False)

    # Master Video Preview Player
    with gr.Row():
        dash_video_player = create_video_preview(label="Live Studio Video Preview (Master Output)")

    # Pipeline Stepper Graph
    with gr.Row():
        pipeline_graph = create_pipeline_graph()

    # Quick Control Buttons
    start_btn, pause_btn, resume_btn, stop_btn, retry_btn = create_control_buttons()

    return {
        "card_res": card_res,
        "card_vision": card_vision,
        "card_style": card_style,
        "video_player": dash_video_player,
        "pipeline_graph": pipeline_graph,
        "start_btn": start_btn,
        "pause_btn": pause_btn,
        "resume_btn": resume_btn,
        "stop_btn": stop_btn,
        "retry_btn": retry_btn,
    }
