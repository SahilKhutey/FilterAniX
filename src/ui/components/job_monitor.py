from __future__ import annotations

import gradio as gr


def create_job_monitor_cards():
    """Renders real-time telemetry indicator boxes."""
    with gr.Row():
        with gr.Column(scale=1):
            status_box = gr.Textbox(label="Status", value="IDLE", interactive=False)
        with gr.Column(scale=2):
            stage_box = gr.Textbox(label="Current Stage", value="Awaiting Input", interactive=False)
        with gr.Column(scale=1):
            progress_box = gr.Textbox(label="Progress", value="0.0%", interactive=False)
        with gr.Column(scale=1):
            fps_box = gr.Textbox(label="Speed", value="0.0 FPS", interactive=False)
        with gr.Column(scale=1):
            eta_box = gr.Textbox(label="ETA", value="00:00", interactive=False)
        with gr.Column(scale=1):
            ram_box = gr.Textbox(label="Memory", value="0.0 GB", interactive=False)

    return status_box, stage_box, progress_box, fps_box, eta_box, ram_box


def create_control_buttons():
    """Renders the standard production execution control buttons."""
    with gr.Row():
        start_btn = gr.Button("⚡ Start Job", variant="primary")
        pause_btn = gr.Button("⏸ Pause", variant="secondary", interactive=False)
        resume_btn = gr.Button("▶ Resume", variant="secondary", interactive=False)
        stop_btn = gr.Button("⛔ Stop", variant="stop", interactive=False)
        retry_btn = gr.Button("🔄 Retry", variant="secondary", interactive=False)

    return start_btn, pause_btn, resume_btn, stop_btn, retry_btn
