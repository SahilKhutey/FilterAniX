"""Mathematical Anime Compositor: Executes canonical image-field sequence."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .color_field import compute_color_field
from .tone_field import compute_tone_field
from .palette_field import compute_palette_projection
from .shadow_field import compute_shadow_field
from .highlight_field import compute_highlight_field
from .edge_field import compute_edge_field
from .face_field import compute_face_mask, compute_semantic_face_masks, apply_face_modulation
from .texture_field import compute_foreground_mask, apply_background_simplification
from .lighting_field import compute_lighting_field
from .temporal_field import TemporalOpticalFlowField


class MathematicalAnimeCompositor:
    """
    Executes the canonical mathematical image-field transformation pipeline:
        I_t -> C_t -> Y_t -> P_t -> S_t -> H_t -> E_t -> V_t -> L_t -> T_t -> A_t
    """

    def __init__(self, style: MathematicalAnimeStyle):
        self.style = style
        self.temporal_field = TemporalOpticalFlowField(self.style)

    def reset(self) -> None:
        """Resets temporal memory across scene cuts or video boundaries."""
        self.temporal_field.reset()

    def transform(
        self,
        rgb_uint8: np.ndarray,
        vision_data: Optional[Any] = None,
        lipsync_record: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        palette_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transforms input frame into stylized anime illustration.
        Returns:
            (output_uint8_rgb, telemetry_dict)
        """
        final_uint8, _, telemetry = self.transform_stages(
            rgb_uint8=rgb_uint8,
            vision_data=vision_data,
            lipsync_record=lipsync_record,
            scene_cut=scene_cut,
            stabilize=stabilize,
            palette_override=palette_override,
        )
        return final_uint8, telemetry

    def transform_stages(
        self,
        rgb_uint8: np.ndarray,
        vision_data: Optional[Any] = None,
        lipsync_record: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        palette_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Executes full canonical field transformation and returns:
            (final_canvas_uint8, stages_dict, telemetry_dict)
        where stages_dict contains uint8 RGB visualizations for MTH-02 through MTH-10.
        """
        h, w, c = rgb_uint8.shape
        source_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

        # 0. Extract Content-Aware Semantic Hierarchy Masks
        primary_face = None
        if vision_data is not None:
            faces = getattr(vision_data, "faces", []) or []
            if isinstance(faces, list) and len(faces) > 0:
                primary_face = faces[0]

        face_mask, hair_mask, eye_mask, mouth_mask, skin_mask = compute_semantic_face_masks(
            h, w, primary_face
        )
        classical_fg = compute_foreground_mask(h, w, vision_data)
        neural_matte = getattr(vision_data, "neural_matte", None)
        if neural_matte is not None and neural_matte.shape[:2] == (h, w):
            # Soft continuous portrait matting fusion (MODNet / U²-Netp)
            fg_mask = np.clip(0.85 * neural_matte + 0.15 * classical_fg, 0.0, 1.0)
        else:
            fg_mask = classical_fg

        clothing_mask = np.clip(fg_mask - face_mask - hair_mask, 0.0, 1.0)
        bg_mask = np.clip(1.0 - fg_mask, 0.0, 1.0)

        # 1. Color Field & Detail Decomposition (C_t / MTH-02)
        color_field, smooth_base = compute_color_field(rgb_uint8, self.style)

        # 2. Anime Luminance & Tone Mapping (Y_t / MTH-03)
        Y_orig, Y_anime, toned_field = compute_tone_field(color_field, self.style)

        # 3. Soft Color Quantization & Palette Projection (P_t / MTH-04)
        palette_field = compute_palette_projection(toned_field, self.style, palette=palette_override)

        # 4. Content-Aware Cel-Shadow Field (S_t / MTH-06 Shadow)
        # Skin receives soft warm ambient shading; eyes receive zero dark crushing; clothing receives crisp cel bands.
        raw_shaded_field, shadow_mask = compute_shadow_field(palette_field, Y_anime, self.style)
        shaded_field = raw_shaded_field.copy()
        if np.max(skin_mask) > 1e-4:
            # Soften shadow depth on face skin to avoid dirty/bruised look
            skin_blend = skin_mask * 0.45
            shaded_field = (1.0 - skin_blend) * shaded_field + skin_blend * palette_field
        if np.max(eye_mask) > 1e-4:
            # Strictly preserve eye catchlights and sclera from dark shadow crushing
            shaded_field = (1.0 - eye_mask) * shaded_field + eye_mask * palette_field

        # 5. Highlight Field (H_t / MTH-06 Highlight)
        lit_field, highlight_mask = compute_highlight_field(shaded_field, Y_anime, self.style)

        # 6. Content-Aware Line-Art & Ink Field (E_t / MTH-05)
        # Background lines are softened to avoid clutter; skin lines are delicate; clothing silhouettes are bold.
        raw_line_field, ink_intensity, raw_edges = compute_edge_field(lit_field, Y_anime, self.style)

        # Compute semantic line attenuation mask:
        # High in background and skin = softer line impact; clothing/silhouette = full line strength
        line_weight = np.ones((h, w, 1), dtype=np.float32)
        if np.max(bg_mask) > 1e-4:
            line_weight -= bg_mask * 0.35  # Subdue background clutter
        if np.max(skin_mask) > 1e-4:
            line_weight -= skin_mask * 0.60  # Avoid whisker-like facial noise
        if np.max(eye_mask) > 1e-4:
            line_weight -= eye_mask * 0.40  # Keep eyes expressive, never a black blob
        line_weight = np.clip(line_weight, 0.2, 1.0)

        # Modulate line field
        line_field = (1.0 - line_weight) * lit_field + line_weight * raw_line_field

        # Depth Field (if provided by Neural Assist Depth Anything V2 Small)
        neural_depth = getattr(vision_data, "neural_depth", None)

        # 7. Cinematic Lighting Field (L_t / MTH-09)
        lit_cinematic, key_light = compute_lighting_field(line_field, Y_anime, self.style)
        if neural_depth is not None and neural_depth.shape[:2] == (h, w):
            depth_factor = 0.88 + 0.12 * neural_depth
            lit_cinematic = np.clip(lit_cinematic * depth_factor, 0.0, 1.0)

        # 8. Background Simplification (G_t / MTH-07)
        if neural_depth is not None and neural_depth.shape[:2] == (h, w):
            depth_bg_mod = np.clip(fg_mask + neural_depth * 0.40, 0.0, 1.0)
            simplified_field = apply_background_simplification(lit_cinematic, depth_bg_mod, self.style)
        else:
            simplified_field = apply_background_simplification(lit_cinematic, fg_mask, self.style)

        # 9. Face & Eye Modulation (V_t / MTH-08)
        face_modulated = apply_face_modulation(
            current_art=simplified_field,
            original_rgb_f=color_field,
            face_mask=face_mask,
            eye_mask=eye_mask,
            hair_mask=hair_mask,
            style=self.style,
        )

        # Information Preservation Guard: Ensure eye/mouth/skin regions never collapse to black
        if np.max(eye_mask) > 1e-4:
            eye_lum = (
                0.299 * face_modulated[:, :, 0]
                + 0.587 * face_modulated[:, :, 1]
                + 0.114 * face_modulated[:, :, 2]
            )[:, :, np.newaxis]
            crushed_eye = (eye_lum < 0.12).astype(np.float32) * eye_mask
            face_modulated = face_modulated + crushed_eye * 0.18

        # 10. Procedural Anime Mouth Rendering (if speech/telemetry present)
        final_canvas_uint8 = np.clip(face_modulated * 255.0, 0.0, 255.0).astype(np.uint8)
        if primary_face is not None:
            viseme = "closed"
            if lipsync_record is not None:
                viseme = getattr(lipsync_record, "viseme", "closed") if not isinstance(lipsync_record, dict) else lipsync_record.get("viseme", "closed")
            from src.art.opencv_renderer import render_anime_mouth
            final_canvas_uint8 = render_anime_mouth(
                canvas_rgb=final_canvas_uint8,
                face_data=primary_face,
                viseme=viseme,
                line_tint=self.style.ink_color,
            )

        # 11. Temporal Regularizer (T_t / MTH-10)
        motion_score = 0.0
        if stabilize:
            final_float = final_canvas_uint8.astype(np.float32) / 255.0
            neural_flow = getattr(vision_data, "neural_flow", None)
            stabilized_float, motion_score = self.temporal_field.stabilize_frame(
                current_art=final_float,
                current_luminance=Y_orig,
                current_source_gray=source_gray,
                scene_cut=scene_cut,
                precomputed_flow=neural_flow,
            )
            final_canvas_uint8 = np.clip(stabilized_float * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            if scene_cut:
                self.reset()

        def _to_u8(f_img: np.ndarray) -> np.ndarray:
            return np.clip(f_img * 255.0, 0.0, 255.0).astype(np.uint8)

        stages: Dict[str, np.ndarray] = {
            "Input": rgb_uint8.copy(),
            "MTH-02 Color": _to_u8(color_field),
            "MTH-03 Tone": _to_u8(toned_field),
            "MTH-04 Palette": _to_u8(palette_field),
            "MTH-05 Edge": _to_u8(line_field),
            "MTH-06 Shadow/Highlight": _to_u8(lit_field),
            "MTH-07 Geometry": _to_u8(simplified_field),
            "MTH-08 Face": _to_u8(face_modulated),
            "MTH-09 Lighting": _to_u8(lit_cinematic),
            "MTH-10 Temporal": final_canvas_uint8.copy(),
            "Final": final_canvas_uint8.copy(),
        }

        telemetry: Dict[str, Any] = {
            "motion_score": motion_score,
            "mean_luminance": float(np.mean(Y_anime)),
            "edge_density": float(np.mean(ink_intensity)),
            "face_detected": primary_face is not None,
            "person_detected": fg_mask is not None and float(np.mean(fg_mask)) > 0.05,
            "mth02_color_mean": float(np.mean(color_field)),
            "mth02_color_min": float(np.min(color_field)),
            "mth02_color_max": float(np.max(color_field)),
            "mth03_tone_mean": float(np.mean(toned_field)),
            "mth04_palette_mean": float(np.mean(palette_field)),
            "mth05_edge_density": float(np.mean(ink_intensity)),
            "mth06_shadow_mean": float(np.mean(shadow_mask)),
            "mth06_highlight_mean": float(np.mean(highlight_mask)),
            "mth08_face_mask": float(np.mean(face_mask)) > 0.001 if face_mask is not None else False,
            "mth08_eye_mask": float(np.mean(eye_mask)) > 0.0001 if eye_mask is not None else False,
            "mth09_lighting_mean": float(np.mean(lit_cinematic)),
            "mth10_temporal_motion": float(motion_score),
        }

        return final_canvas_uint8, stages, telemetry
