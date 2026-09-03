from __future__ import annotations

from typing import Any, Dict
import gradio as gr
from src.art.mathematical import MathematicalAnimeStyle


def create_mathematical_parameter_panel():
    """
    Renders the complete, transparent Mathematical Engine control panel.
    Every slider directly maps to a deterministic parameter in MathematicalAnimeStyle.
    """
    default_style = MathematicalAnimeStyle.creator_anime()

    with gr.Accordion("⚙ Mathematical Engine Parameters (Transparent Tuning)", open=False):
        with gr.Tabs():
            # GLOBAL TONE
            with gr.TabItem("Global Tone"):
                with gr.Row():
                    contrast_slider = gr.Slider(
                        minimum=0.5, maximum=1.8, step=0.01,
                        value=default_style.contrast, label="Contrast Boost"
                    )
                    gamma_slider = gr.Slider(
                        minimum=0.5, maximum=1.5, step=0.01,
                        value=default_style.gamma, label="Gamma"
                    )
                    tone_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.tone_strength, label="Tone Strength"
                    )

            # COLOR FIELD (MTH-02 & MTH-04)
            with gr.TabItem("Color & Palette"):
                with gr.Row():
                    saturation_slider = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.01,
                        value=default_style.saturation, label="Saturation"
                    )
                    palette_mix_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.palette_mix, label="Palette Mix"
                    )
                    color_levels_slider = gr.Slider(
                        minimum=4, maximum=32, step=1,
                        value=default_style.color_levels, label="Color Levels (Quantization)"
                    )
                    palette_temp_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.palette_temperature, label="Palette Temperature"
                    )

            # TONE FIELD (MTH-03)
            with gr.TabItem("Tone & Smoothing"):
                with gr.Row():
                    smooth_sigma_slider = gr.Slider(
                        minimum=0.0, maximum=4.0, step=0.05,
                        value=default_style.smooth_sigma, label="Smooth Sigma"
                    )
                    texture_suppress_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.texture_suppression, label="Texture Suppression"
                    )
                    detail_retention_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.detail_retention, label="Detail Retention"
                    )

            # EDGE FIELD (MTH-05)
            with gr.TabItem("Edge & Ink Lines"):
                with gr.Row():
                    edge_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.edge_strength, label="Edge Strength"
                    )
                    edge_threshold_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.edge_threshold, label="Edge Threshold"
                    )
                    edge_softness_slider = gr.Slider(
                        minimum=0.01, maximum=0.3, step=0.005,
                        value=default_style.edge_softness, label="Edge Softness"
                    )
                    line_darkness_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.line_darkness, label="Line Darkness"
                    )

            # LIGHTING & SHADOWS (MTH-06 & MTH-09)
            with gr.TabItem("Lighting & Cel Shading"):
                with gr.Row():
                    shadow_thresh_slider = gr.Slider(
                        minimum=0.1, maximum=0.8, step=0.01,
                        value=default_style.shadow_threshold, label="Shadow Threshold"
                    )
                    shadow_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.shadow_strength, label="Shadow Strength"
                    )
                    highlight_thresh_slider = gr.Slider(
                        minimum=0.5, maximum=0.99, step=0.01,
                        value=default_style.highlight_threshold, label="Highlight Threshold"
                    )
                    highlight_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.highlight_strength, label="Highlight Strength"
                    )
                    warm_light_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.warm_light_strength, label="Warm Lighting"
                    )

            # CHARACTER & GEOMETRY (MTH-07 & MTH-08)
            with gr.TabItem("Geometry & Face"):
                with gr.Row():
                    face_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.face_geometry_strength, label="Face Feature Strength"
                    )
                    pose_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.pose_geometry_strength, label="Pose Strength"
                    )
                    hand_strength_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.hand_geometry_strength, label="Hand Strength"
                    )
                    bg_simplify_slider = gr.Slider(
                        minimum=0.0, maximum=1.0, step=0.01,
                        value=default_style.background_simplification, label="Background Simplification"
                    )

            # TEMPORAL (MTH-10)
            with gr.TabItem("Temporal Stability"):
                with gr.Row():
                    temporal_strength_slider = gr.Slider(
                        minimum=0.0, maximum=0.5, step=0.01,
                        value=default_style.temporal_strength, label="Temporal Blend Strength"
                    )
                    motion_limit_slider = gr.Slider(
                        minimum=0.01, maximum=0.5, step=0.01,
                        value=default_style.temporal_motion_limit, label="Motion Threshold Limit"
                    )

    controls_list = [
        contrast_slider,
        gamma_slider,
        tone_strength_slider,
        saturation_slider,
        palette_mix_slider,
        color_levels_slider,
        palette_temp_slider,
        smooth_sigma_slider,
        texture_suppress_slider,
        detail_retention_slider,
        edge_strength_slider,
        edge_threshold_slider,
        edge_softness_slider,
        line_darkness_slider,
        shadow_thresh_slider,
        shadow_strength_slider,
        highlight_thresh_slider,
        highlight_strength_slider,
        warm_light_slider,
        face_strength_slider,
        pose_strength_slider,
        hand_strength_slider,
        bg_simplify_slider,
        temporal_strength_slider,
        motion_limit_slider,
    ]

    return controls_list


def build_math_params_dict(
    contrast: float,
    gamma: float,
    tone_strength: float,
    saturation: float,
    palette_mix: float,
    color_levels: int,
    palette_temp: float,
    smooth_sigma: float,
    texture_suppress: float,
    detail_retention: float,
    edge_strength: float,
    edge_threshold: float,
    edge_softness: float,
    line_darkness: float,
    shadow_thresh: float,
    shadow_strength: float,
    highlight_thresh: float,
    highlight_strength: float,
    warm_light: float,
    face_strength: float,
    pose_strength: float,
    hand_strength: float,
    bg_simplify: float,
    temporal_strength: float,
    motion_limit: float,
) -> Dict[str, Any]:
    """Assembles raw slider values into a typed dict for MathematicalAnimeStyle."""
    return {
        "contrast": float(contrast),
        "gamma": float(gamma),
        "tone_strength": float(tone_strength),
        "saturation": float(saturation),
        "palette_mix": float(palette_mix),
        "color_levels": int(color_levels),
        "palette_temperature": float(palette_temp),
        "smooth_sigma": float(smooth_sigma),
        "texture_suppression": float(texture_suppress),
        "detail_retention": float(detail_retention),
        "edge_strength": float(edge_strength),
        "edge_threshold": float(edge_threshold),
        "edge_softness": float(edge_softness),
        "line_darkness": float(line_darkness),
        "shadow_threshold": float(shadow_thresh),
        "shadow_strength": float(shadow_strength),
        "highlight_threshold": float(highlight_thresh),
        "highlight_strength": float(highlight_strength),
        "warm_light_strength": float(warm_light),
        "face_geometry_strength": float(face_strength),
        "pose_geometry_strength": float(pose_strength),
        "hand_geometry_strength": float(hand_strength),
        "background_simplification": float(bg_simplify),
        "temporal_strength": float(temporal_strength),
        "temporal_motion_limit": float(motion_limit),
    }
