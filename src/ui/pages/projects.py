from __future__ import annotations

import gradio as gr
from src.ui.callbacks import list_recent_projects


def render_projects_page():
    """Renders the Project Management & Resumability screen."""
    gr.Markdown("### 📁 Project Workspace Manager")
    gr.Markdown("FilterAniX projects preserve granular stage artifacts (`source`, `vision`, `consistency`, `artistic`, `lipsync`, `output`).")

    with gr.Row():
        refresh_proj_btn = gr.Button("🔄 Refresh Projects List", variant="secondary")

    projects_table = gr.Dataframe(
        headers=["Project Name", "Status", "Stage Progress", "Last Updated"],
        value=list_recent_projects,
        interactive=False,
    )

    with gr.Row():
        selected_project_name = gr.Textbox(label="Selected Project", placeholder="Enter or select project name")

    with gr.Row():
        open_btn = gr.Button("📂 Load Project", variant="secondary")
        resume_btn = gr.Button("▶ Resume Cached Pipeline", variant="primary")
        delete_btn = gr.Button("🗑 Delete Project", variant="stop")

    project_action_msg = gr.Textbox(label="Project Action Result", lines=2, interactive=False)

    return {
        "refresh_btn": refresh_proj_btn,
        "table": projects_table,
        "selected_project": selected_project_name,
        "open_btn": open_btn,
        "resume_btn": resume_btn,
        "delete_btn": delete_btn,
        "action_msg": project_action_msg,
    }
