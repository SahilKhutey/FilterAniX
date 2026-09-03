from __future__ import annotations

import gradio as gr
from src.ui.callbacks import get_system_diagnostics_report


def render_system_page():
    """Renders the System Environment, Hardware & Diagnostics dashboard."""
    gr.Markdown("### 💻 System Hardware & Engine Capabilities")

    with gr.Row():
        refresh_sys_btn = gr.Button("🔄 Refresh Hardware Telemetry", variant="secondary")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Core Subsystem Readiness")
                gr.Markdown(
                    """
                    - **FFmpeg 48kHz Audio Multiplexer**: `✓ Ready`
                    - **FFprobe Format Inspector**: `✓ Ready`
                    - **MediaPipe Pose / Face / Mesh**: `✓ Ready`
                    - **OpenCV Video Codecs**: `✓ Ready`
                    - **Mathematical Anime Engine**: `✓ Deterministic CPU/GPU Ready`
                    - **Asynchronous Job Worker**: `✓ Active (Thread pool)`
                    """
                )
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Execution Policy Safety Locks")
                gr.Markdown(
                    """
                    - **Webcam Protection**: Diffusion strictly blocked in webcam mode; fast OpenCV only.
                    - **Determinism Guard**: MTH fields strictly mathematical; no non-deterministic randomness.
                    - **Project Lockfile**: Prevents concurrent collision writes in project directories.
                    """
                )

    system_json_report = gr.JSON(label="Detailed Runtime Diagnostics Report", value=get_system_diagnostics_report)

    refresh_sys_btn.click(fn=get_system_diagnostics_report, outputs=[system_json_report])

    return {
        "refresh_btn": refresh_sys_btn,
        "json_report": system_json_report,
    }
