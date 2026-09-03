from __future__ import annotations

import gradio as gr


def create_status_bar() -> gr.Textbox:
    """Renders the persistent bottom telemetry status bar."""
    status_bar = gr.Textbox(
        value="● SYSTEM READY   |   FilterAniX Studio v2.0   |   No active render job",
        interactive=False,
        show_label=False,
        elem_classes=["studio-status-bar"],
        lines=1,
    )
    return status_bar
