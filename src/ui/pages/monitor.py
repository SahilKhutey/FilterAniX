from __future__ import annotations

import gradio as gr
from src.ui.components.job_monitor import create_job_monitor_cards, create_control_buttons
from src.ui.components.pipeline_graph import create_pipeline_graph
from src.ui.components.notifications import create_notifications_feed


def render_monitor_page():
    """Renders the Production Pipeline Monitor & Error Recovery interface."""
    gr.Markdown("### 📊 Production Pipeline Monitor & Live Telemetry")

    # Real-time Telemetry Cards (Status, Stage, Progress %, Speed, ETA, Memory)
    status_box, stage_box, progress_box, fps_box, eta_box, ram_box = create_job_monitor_cards()

    # Pipeline Stepper & Notifications
    with gr.Row():
        with gr.Column(scale=3):
            pipeline_stepper = create_pipeline_graph()
        with gr.Column(scale=2):
            notif_feed = create_notifications_feed()
            log_viewer = gr.Textbox(label="Stage Execution Log", lines=8, interactive=False)

    # Control Buttons
    start_btn, pause_btn, resume_btn, stop_btn, retry_btn = create_control_buttons()

    # Production Error & Recovery UI
    with gr.Accordion("⚠ Rendering Incident & Recovery Console", open=False, elem_classes=["error-panel"]):
        gr.Markdown(
            """
            #### ⚠ Rendering Incident Detected
            If a frame fails or encounters unexpected input anomalies, the incident is captured below with technical diagnostics.
            """
        )
        with gr.Row():
            err_stage = gr.Textbox(label="Failed Stage", value="None", interactive=False)
            err_frame = gr.Textbox(label="Failed Frame", value="None", interactive=False)
        err_msg = gr.Textbox(label="Error Summary", value="No active errors.", lines=2, interactive=False)

        with gr.Accordion("Technical Traceback & Memory State", open=False):
            err_traceback = gr.Textbox(label="Traceback Details", lines=6, interactive=False)

        with gr.Row():
            recover_retry_btn = gr.Button("🔄 Retry Stage", variant="primary")
            recover_skip_btn = gr.Button("⏭ Skip Corrupt Frame", variant="secondary")
            recover_stop_btn = gr.Button("⛔ Terminate Job", variant="stop")

    return {
        "status_box": status_box,
        "stage_box": stage_box,
        "progress_box": progress_box,
        "fps_box": fps_box,
        "eta_box": eta_box,
        "ram_box": ram_box,
        "pipeline_stepper": pipeline_stepper,
        "notif_feed": notif_feed,
        "log_viewer": log_viewer,
        "start_btn": start_btn,
        "pause_btn": pause_btn,
        "resume_btn": resume_btn,
        "stop_btn": stop_btn,
        "retry_btn": retry_btn,
        "err_stage": err_stage,
        "err_frame": err_frame,
        "err_msg": err_msg,
        "err_traceback": err_traceback,
        "recover_retry_btn": recover_retry_btn,
        "recover_skip_btn": recover_skip_btn,
        "recover_stop_btn": recover_stop_btn,
    }
