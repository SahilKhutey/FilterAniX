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
from .face_field import compute_face_mask, apply_face_modulation
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
        h, w, c = rgb_uint8.shape
        source_gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)

        # 1. Color Field & Detail Decomposition (C_t)
        color_field, smooth_base = compute_color_field(rgb_uint8, self.style)

        # 2. Anime Luminance & Tone Mapping (Y_t)
        Y_orig, Y_anime, toned_field = compute_tone_field(color_field, self.style)

        # 3. Soft Color Quantization & Palette Projection (P_t)
        palette_field = compute_palette_projection(toned_field, self.style, palette=palette_override)

        # 4. Cel-Shadow Field (S_t)
        shaded_field, shadow_mask = compute_shadow_field(palette_field, Y_anime, self.style)

        # 5. Highlight Field (H_t)
        lit_field, highlight_mask = compute_highlight_field(shaded_field, Y_anime, self.style)

        # 6. Anime Line-Art & Ink Field (E_t)
        line_field, ink_intensity, raw_edges = compute_edge_field(lit_field, Y_anime, self.style)

        # 7. Cinematic Lighting Field (L_t)
        lit_cinematic, key_light = compute_lighting_field(line_field, Y_anime, self.style)

        # 8. Background vs Foreground Simplification (Texture Suppression)
        fg_mask = compute_foreground_mask(h, w, vision_data)
        simplified_field = apply_background_simplification(lit_cinematic, fg_mask, self.style)

        # 9. Face & Eye Modulation (V_t / Face Field)
        primary_face = None
        if vision_data is not None:
            faces = getattr(vision_data, "faces", []) or []
            if isinstance(faces, list) and len(faces) > 0:
                primary_face = faces[0]

        face_mask, hair_mask, eye_mask = compute_face_mask(h, w, primary_face)
        face_modulated = apply_face_modulation(
            current_art=simplified_field,
            original_rgb_f=color_field,
            face_mask=face_mask,
            eye_mask=eye_mask,
            hair_mask=hair_mask,
            style=self.style,
        )

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

        # 11. Temporal Regularizer (T_t -> A_t)
        motion_score = 0.0
        if stabilize:
            final_float = final_canvas_uint8.astype(np.float32) / 255.0
            stabilized_float, motion_score = self.temporal_field.stabilize_frame(
                current_art=final_float,
                current_luminance=Y_orig,
                current_source_gray=source_gray,
                scene_cut=scene_cut,
            )
            final_canvas_uint8 = np.clip(stabilized_float * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            if scene_cut:
                self.reset()

        telemetry = {
            "motion_score": motion_score,
            "mean_luminance": float(np.mean(Y_anime)),
            "edge_density": float(np.mean(ink_intensity)),
            "face_detected": primary_face is not None,
        }

        return final_canvas_uint8, telemetry
