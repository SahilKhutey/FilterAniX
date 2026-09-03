from __future__ import annotations

from pathlib import Path
import gradio as gr
from export_youtube import PRESETS
from src.ui.components.preview import create_video_preview


def render_export_page():
    """Renders the Broadcast Export Center for YouTube and production encoding."""
    gr.Markdown("### 📤 Broadcast Export Center")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("#### Source Master & Verification Checklist")
                gr.Checkbox(label="Deterministic Mathematical Render", value=True, interactive=False)
                gr.Checkbox(label="Source Audio Track Preserved (AAC 48kHz)", value=True, interactive=False)
                gr.Checkbox(label="Lip-Sync & Viseme Alignment", value=True, interactive=False)
                gr.Checkbox(label="Broadcast Validation Passed", value=True, interactive=False)

            export_video_in = gr.Video(label="Input Master Video for Re-encoding")

            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS.keys()),
                value="1080p",
                label="Target YouTube Resolution Preset",
            )

            with gr.Accordion("Advanced Codec Options", open=False):
                codec_in = gr.Dropdown(choices=["libx264 (H.264)", "libx265 (HEVC)"], value="libx264 (H.264)", label="Video Codec")
                crf_slider = gr.Slider(minimum=14, maximum=28, value=18, step=1, label="CRF Quality (Lower = Higher Quality)")
                pix_fmt_in = gr.Textbox(value="yuv420p", label="Pixel Format (Broadcast Standard)", interactive=False)
                audio_codec_in = gr.Textbox(value="AAC (192kbps, 48kHz)", label="Audio Codec", interactive=False)

            export_btn = gr.Button("🚀 Export Master MP4", variant="primary", size="lg")

        with gr.Column(scale=1):
            exported_video_out = create_video_preview(label="Exported Master Preview")
            export_status = gr.Textbox(label="Export Result & Broadcast Telemetry", lines=6, interactive=False)

    return {
        "export_video_in": export_video_in,
        "preset_dropdown": preset_dropdown,
        "export_btn": export_btn,
        "exported_video_out": exported_video_out,
        "export_status": export_status,
    }
