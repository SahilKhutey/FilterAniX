from __future__ import annotations

import gradio as gr


def create_video_preview(label: str = "Production Video Master") -> gr.Video:
    """Renders a standard video playback preview widget."""
    return gr.Video(
        label=label,
        interactive=False,
    )


def create_split_frame_preview() -> tuple[gr.Image, gr.Image, gr.Image]:
    """Renders interactive original, stylized, and split comparison preview images."""
    with gr.Row():
        orig_img = gr.Image(label="Original Creator Frame", interactive=False)
        split_img = gr.Image(label="Split Comparison (Original | Render)", interactive=False)
        art_img = gr.Image(label="Mathematical Render Result", interactive=False)
    return orig_img, split_img, art_img
