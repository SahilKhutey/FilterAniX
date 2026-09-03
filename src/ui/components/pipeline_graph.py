from __future__ import annotations

import gradio as gr
from src.ui.telemetry import format_pipeline_stepper


def create_pipeline_graph() -> gr.Textbox:
    """Renders the live pipeline stepper showing all 8 phases and MTH-02..10 sub-stages."""
    default_text = format_pipeline_stepper({"status": "idle", "stage": "", "progress": 0.0})
    pipeline_display = gr.Textbox(
        value=default_text,
        label="Pipeline Execution & Mathematical Sub-stages",
        lines=16,
        interactive=False,
        elem_classes=["pipeline-stepper"],
    )
    return pipeline_display
