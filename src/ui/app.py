from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np

from src.core.jobs import JobManager
from src.core.hardware import system_info
from src.ui.theme import get_studio_theme, STUDIO_CUSTOM_CSS
from src.ui.components.header import create_header
from src.ui.components.status_bar import create_status_bar
from src.ui.components.notifications import format_notification_text
from src.ui.components.parameter_panel import build_math_params_dict
from src.ui.telemetry import (
    get_job_state,
    format_status_bar,
    format_pipeline_stepper,
    get_current_memory_gb,
)
from src.ui.callbacks import (
    inspect_input_video_file,
    start_pipeline_job,
    pause_active_job,
    resume_active_job,
    stop_active_job,
    retry_failed_job,
    export_preset_video,
    render_style_lab_frame,
    get_style_lab_stage_frame,
    get_style_lab_diagnostics,
    get_neural_telemetry_report,
    save_style_preset,
    list_recent_projects,
)
from src.ui.state import UIState, ui_state_manager
from src.ui.pages.dashboard import render_dashboard_page
from src.ui.pages.create import render_create_page
from src.ui.pages.projects import render_projects_page
from src.ui.pages.live import render_live_page
from src.ui.pages.style_lab import render_style_lab_page
from src.ui.pages.monitor import render_monitor_page
from src.ui.pages.export import render_export_page
from src.ui.pages.system import render_system_page


# Global studio job manager singleton
studio_job_manager = JobManager(max_workers=1)


def create_app() -> gr.Blocks:
    """Builds and wires FilterAniX Studio UI v2.0."""
    with gr.Blocks(
        title="FilterAniX Studio v2.0",
    ) as demo:

        # Header
        create_header()

        # Active Job ID state
        job_id_state = gr.State("")

        with gr.Tabs() as main_tabs:
            # 1. DASHBOARD
            with gr.TabItem("📊 Dashboard", id="tab_dashboard"):
                dash_page = render_dashboard_page()

            # 2. CREATE
            with gr.TabItem("🎬 Create", id="tab_create"):
                create_page = render_create_page()

            # 3. PROJECTS
            with gr.TabItem("📁 Projects", id="tab_projects"):
                projects_page = render_projects_page()

            # 4. LIVE
            with gr.TabItem("📹 Live", id="tab_live"):
                live_page = render_live_page()

            # 5. STYLE LAB
            with gr.TabItem("🎨 Style Lab", id="tab_style_lab"):
                style_lab_page = render_style_lab_page()

            # 6. PIPELINE MONITOR
            with gr.TabItem("⚙ Pipeline Monitor", id="tab_monitor"):
                monitor_page = render_monitor_page()

            # 7. EXPORT
            with gr.TabItem("📤 Export", id="tab_export"):
                export_page = render_export_page()

            # 8. SYSTEM
            with gr.TabItem("💻 System", id="tab_system"):
                system_page = render_system_page()

        # Bottom persistent status bar
        status_bar = create_status_bar()

        # =================================================================
        # CALLBACK WIRING
        # =================================================================

        # --- CREATE PAGE WIRING ---
        def on_inspect_video(video_path):
            res, fps, frames, dur, audio = inspect_input_video_file(video_path)
            # Also populate dashboard cards
            dash_card_res = f"{res} ({dur})" if res != "-" else "No input"
            dash_card_vis = f"{frames} @ {fps}" if frames != "-" else "Awaiting Video"
            return res, fps, frames, dur, audio, dash_card_res, dash_card_vis

        create_page["inspect_btn"].click(
            fn=on_inspect_video,
            inputs=[create_page["video_input"]],
            outputs=[
                create_page["meta_res"],
                create_page["meta_fps"],
                create_page["meta_frames"],
                create_page["meta_dur"],
                create_page["meta_audio"],
                dash_page["card_res"],
                dash_page["card_vision"],
            ],
        )

        # Start job from Create page
        def on_create_start(video, name, style, *math_args):
            math_dict = build_math_params_dict(*math_args)
            j_id, msg, btn_st = start_pipeline_job(
                studio_job_manager,
                video,
                name,
                style,
                math_dict,
            )
            pause_upd = gr.update(interactive=btn_st["pause_enabled"])
            stop_upd = gr.update(interactive=btn_st["stop_enabled"])
            resume_upd = gr.update(interactive=btn_st["resume_enabled"])
            return j_id, msg, pause_upd, stop_upd, resume_upd

        create_page["start_btn"].click(
            fn=on_create_start,
            inputs=[
                create_page["video_input"],
                create_page["project_name"],
                create_page["style_dropdown"],
                *create_page["math_controls"],
            ],
            outputs=[
                job_id_state,
                create_page["status_msg"],
                dash_page["pause_btn"],
                dash_page["stop_btn"],
                dash_page["resume_btn"],
            ],
        )

        # Start job from Dashboard page
        def on_dash_start(video, name, style):
            j_id, msg, btn_st = start_pipeline_job(
                studio_job_manager,
                video,
                name,
                style,
            )
            pause_upd = gr.update(interactive=btn_st["pause_enabled"])
            stop_upd = gr.update(interactive=btn_st["stop_enabled"])
            resume_upd = gr.update(interactive=btn_st["resume_enabled"])
            return j_id, pause_upd, stop_upd, resume_upd

        dash_page["start_btn"].click(
            fn=on_dash_start,
            inputs=[
                create_page["video_input"],
                create_page["project_name"],
                create_page["style_dropdown"],
            ],
            outputs=[
                job_id_state,
                dash_page["pause_btn"],
                dash_page["stop_btn"],
                dash_page["resume_btn"],
            ],
        )

        # Start job from Monitor page
        monitor_page["start_btn"].click(
            fn=on_dash_start,
            inputs=[
                create_page["video_input"],
                create_page["project_name"],
                create_page["style_dropdown"],
            ],
            outputs=[
                job_id_state,
                dash_page["pause_btn"],
                dash_page["stop_btn"],
                dash_page["resume_btn"],
            ],
        )

        # --- CONTROL BUTTONS WIRING ---
        def on_pause(j_id):
            msg, btn_st = pause_active_job(studio_job_manager, j_id)
            return gr.update(interactive=btn_st["resume_enabled"]), gr.update(interactive=btn_st["pause_enabled"])

        for p_btn in [dash_page["pause_btn"], monitor_page["pause_btn"]]:
            p_btn.click(
                fn=on_pause,
                inputs=[job_id_state],
                outputs=[dash_page["resume_btn"], dash_page["pause_btn"]],
            )

        def on_resume(j_id):
            msg, btn_st = resume_active_job(studio_job_manager, j_id)
            return gr.update(interactive=btn_st["pause_enabled"]), gr.update(interactive=btn_st["resume_enabled"])

        for r_btn in [dash_page["resume_btn"], monitor_page["resume_btn"]]:
            r_btn.click(
                fn=on_resume,
                inputs=[job_id_state],
                outputs=[dash_page["pause_btn"], dash_page["resume_btn"]],
            )

        def on_stop(j_id):
            msg, btn_st = stop_active_job(studio_job_manager, j_id)
            return gr.update(interactive=btn_st["start_enabled"]), gr.update(interactive=False), gr.update(interactive=False)

        for s_btn in [dash_page["stop_btn"], monitor_page["stop_btn"]]:
            s_btn.click(
                fn=on_stop,
                inputs=[job_id_state],
                outputs=[dash_page["start_btn"], dash_page["pause_btn"], dash_page["resume_btn"]],
            )

        # --- STYLE LAB WIRING ---
        def on_apply_style_lab(video, frame_idx, enable_neural, *math_args):
            math_dict = build_math_params_dict(*math_args)
            math_dict["enable_neural_assist"] = bool(enable_neural)
            orig, art, split, status = render_style_lab_frame(video, frame_idx, math_dict)
            diag_text = get_style_lab_diagnostics()
            if enable_neural:
                diag_text = f"{diag_text}\n\n=== NEURAL ASSISTANCE TELEMETRY ===\n{get_neural_telemetry_report()}"
            return orig, split, art, status, "Final", diag_text

        style_lab_page["apply_btn"].click(
            fn=on_apply_style_lab,
            inputs=[
                create_page["video_input"],
                style_lab_page["frame_slider"],
                style_lab_page["neural_assist_toggle"],
                *style_lab_page["math_controls"],
            ],
            outputs=[
                style_lab_page["orig_img"],
                style_lab_page["split_img"],
                style_lab_page["art_img"],
                style_lab_page["lab_status"],
                style_lab_page["stage_selector"],
                style_lab_page["engine_diag"],
            ],
        )

        def on_stage_selected(stage_name):
            stage_img = get_style_lab_stage_frame(stage_name)
            if stage_img is not None:
                return stage_img
            return gr.update()

        style_lab_page["stage_selector"].change(
            fn=on_stage_selected,
            inputs=[style_lab_page["stage_selector"]],
            outputs=[style_lab_page["art_img"]],
        )

        def on_prev_frame(idx):
            return max(0, int(idx) - 1)

        def on_next_frame(idx):
            return int(idx) + 1

        style_lab_page["prev_btn"].click(fn=on_prev_frame, inputs=[style_lab_page["frame_slider"]], outputs=[style_lab_page["frame_slider"]])
        style_lab_page["next_btn"].click(fn=on_next_frame, inputs=[style_lab_page["frame_slider"]], outputs=[style_lab_page["frame_slider"]])

        def on_save_preset(name, *math_args):
            math_dict = build_math_params_dict(*math_args)
            return save_style_preset(name, math_dict)

        style_lab_page["save_preset_btn"].click(
            fn=on_save_preset,
            inputs=[style_lab_page["preset_name"], *style_lab_page["math_controls"]],
            outputs=[style_lab_page["save_msg"]],
        )

        style_lab_page["accept_btn"].click(
            fn=lambda: "✔ Accepted: Background simplification updated to 0.58 in active session.",
            outputs=[style_lab_page["suggestion_msg"]],
        )
        style_lab_page["reject_btn"].click(
            fn=lambda: "✖ Rejected: Retained current background simplification (0.65).",
            outputs=[style_lab_page["suggestion_msg"]],
        )

        # --- PROJECTS PAGE WIRING ---
        projects_page["refresh_btn"].click(
            fn=list_recent_projects,
            outputs=[projects_page["table"]],
        )

        def on_resume_project(project_name, style):
            if not project_name:
                return "", "Please select a project to resume."
            proj_dir = Path("projects") / project_name
            if not proj_dir.exists():
                return "", f"Project {project_name} not found."

            source_files = list((proj_dir / "source").glob("*.*"))
            if not source_files:
                return "", f"No source video found in {proj_dir / 'source'}."

            return start_pipeline_job(
                studio_job_manager,
                str(source_files[0]),
                project_name,
                style,
            )[:2]

        projects_page["resume_btn"].click(
            fn=on_resume_project,
            inputs=[projects_page["selected_project"], create_page["style_dropdown"]],
            outputs=[job_id_state, projects_page["action_msg"]],
        )

        def on_delete_project(project_name):
            if not project_name:
                return "Please enter a project name to delete."
            proj_dir = Path("projects") / project_name
            if proj_dir.exists() and proj_dir.is_dir():
                import shutil
                try:
                    shutil.rmtree(proj_dir)
                    return f"Deleted project: {project_name}"
                except Exception as exc:
                    return f"Failed to delete: {exc}"
            return "Project not found."

        projects_page["delete_btn"].click(
            fn=on_delete_project,
            inputs=[projects_page["selected_project"]],
            outputs=[projects_page["action_msg"]],
        )

        # --- EXPORT PAGE WIRING ---
        export_page["export_btn"].click(
            fn=export_preset_video,
            inputs=[export_page["export_video_in"], export_page["preset_dropdown"]],
            outputs=[export_page["exported_video_out"], export_page["export_status"]],
        )

        # --- LIVE RECURRING TELEMETRY POLLER (500ms) ---
        telemetry_timer = gr.Timer(value=0.5, active=True)

        def on_telemetry_tick(j_id):
            snap = get_job_state(studio_job_manager, j_id)
            sb_text = format_status_bar(snap)
            pipe_text = format_pipeline_stepper(snap)
            notifs_text = format_notification_text()

            # Check for final video
            final_vid = None
            if snap.get("status") == "complete":
                res = snap.get("result")
                if isinstance(res, dict) and "final_video" in res:
                    final_vid = res["final_video"]
                elif snap.get("current_output"):
                    final_vid = snap.get("current_output")

            status_str = snap.get("status", "idle").upper()
            stage_str = snap.get("stage", "Awaiting Input")
            if snap.get("substage"):
                stage_str += f" ({snap.get('substage')})"

            prog_str = f"{snap.get('progress', 0.0) * 100:.1f}%"
            fps_str = f"{snap.get('fps', 0.0):.1f} FPS"
            eta_val = snap.get("eta_seconds", 0.0)
            eta_str = f"{int(eta_val // 60):02d}:{int(eta_val % 60):02d}"
            ram_str = f"{get_current_memory_gb():.1f} GB"
            msg_str = snap.get("message", "")

            # Button interactivity states
            btn_st = ui_state_manager.get_button_states()
            p_upd = gr.update(interactive=btn_st["pause_enabled"])
            r_upd = gr.update(interactive=btn_st["resume_enabled"])
            s_upd = gr.update(interactive=btn_st["stop_enabled"])

            # Error info if failed
            err_st = snap.get("stage", "None") if snap.get("status") == "failed" else "None"
            err_f = str(snap.get("frame", "None")) if snap.get("status") == "failed" else "None"
            err_m = snap.get("error", "No active errors.") or "No active errors."

            return (
                sb_text,
                pipe_text,
                pipe_text,  # for monitor page
                notifs_text,
                status_str,
                stage_str,
                prog_str,
                fps_str,
                eta_str,
                ram_str,
                msg_str,
                final_vid,
                p_upd,
                r_upd,
                s_upd,
                err_st,
                err_f,
                err_m,
            )

        telemetry_timer.tick(
            fn=on_telemetry_tick,
            inputs=[job_id_state],
            outputs=[
                status_bar,
                dash_page["pipeline_graph"],
                monitor_page["pipeline_stepper"],
                monitor_page["notif_feed"],
                monitor_page["status_box"],
                monitor_page["stage_box"],
                monitor_page["progress_box"],
                monitor_page["fps_box"],
                monitor_page["eta_box"],
                monitor_page["ram_box"],
                monitor_page["log_viewer"],
                dash_page["video_player"],
                dash_page["pause_btn"],
                dash_page["resume_btn"],
                dash_page["stop_btn"],
                monitor_page["err_stage"],
                monitor_page["err_frame"],
                monitor_page["err_msg"],
            ],
        )

    return demo
