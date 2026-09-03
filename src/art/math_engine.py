from __future__ import annotations

from typing import Any, Optional, Tuple
import cv2
import numpy as np

from src.art.types import StyleConfig, StylePreset


# Default 6-tier continuous anime color palette (Light, Light-Mid, Mid, Dark-Mid, Dark, Deep-Dark)
DEFAULT_ANIME_PALETTE = np.array([
    [255, 246, 235],  # Light / Specular
    [245, 218, 192],  # Light-Mid / Warm Skin Diffuse
    [212, 168, 138],  # Mid / Amber Ambient
    [160, 115, 95],   # Dark-Mid / Warm Shadow
    [92, 65, 78],     # Dark / Deep Cel Shadow
    [32, 24, 35],     # Deep-Dark / Ink
], dtype=np.float32) / 255.0


class MathematicalStyleEngine:
    """
    Deterministic Mathematical Video-to-Style Transformation Engine.
    Transforms every input pixel through a composable mathematical operator pipeline:
        A_t = C_t ∘ E_t ∘ G_t ∘ Q_t ∘ L_t (I_t)
    with continuous temporal regularization:
        A_t' = (1 - λ_t) A_t + λ_t A_{t-1}
    """

    def __init__(self, config: Optional[StyleConfig] = None):
        self.config = config or StyleConfig()
        self.palette = DEFAULT_ANIME_PALETTE.copy()
        self._prev_luminance: Optional[np.ndarray] = None
        self._prev_art: Optional[np.ndarray] = None

    def reset_temporal(self) -> None:
        """Resets temporal memory across scene cuts or hard boundaries."""
        self._prev_luminance = None
        self._prev_art = None

    def render(
        self,
        rgb: np.ndarray,
        vision_data: Optional[Any] = None,
        lipsync_record: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        reference_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Executes full deterministic mathematical transformation on a single frame:
        Returns uint8 RGB array with identical spatial dimensions (W x H x 3).
        """
        if scene_cut:
            self.reset_temporal()

        h, w = rgb.shape[:2]

        # ---------------------------------------------------------
        # Step A: Normalize to [0.0, 1.0] float32
        # ---------------------------------------------------------
        img_f = rgb.astype(np.float32) / 255.0

        # ---------------------------------------------------------
        # Step B: Local Image Field & Detail Decomposition
        # ---------------------------------------------------------
        # Bilateral / edge-preserving smoothing removes photographic micro-noise while keeping edges crisp
        smooth_rgb = cv2.bilateralFilter(
            (img_f * 255.0).astype(np.uint8),
            d=7,
            sigmaColor=40,
            sigmaSpace=40,
        ).astype(np.float32) / 255.0

        detail_field = img_f - smooth_rgb
        base_field = np.clip(
            smooth_rgb + self.config.detail_strength * detail_field,
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Step C: Mathematical Tone & Luminance Transformation
        # ---------------------------------------------------------
        # Standard Rec.601/709 Luminance: Y = 0.299R + 0.587G + 0.114B
        Y = 0.299 * base_field[:, :, 0] + 0.587 * base_field[:, :, 1] + 0.114 * base_field[:, :, 2]

        # Contrast S-curve: Y_c = clip((Y - 0.5) * C + 0.5)
        Y_c = np.clip((Y - 0.5) * self.config.tone_contrast + 0.5, 0.0, 1.0)

        # Gamma transformation: Y_g = Y_c^gamma
        Y_g = np.power(np.maximum(Y_c, 1e-6), self.config.tone_gamma)

        # Tonal modulation ratio
        tone_ratio = (Y_g / np.maximum(Y, 1e-5))[:, :, np.newaxis]
        toned_field = np.clip(
            base_field * (1.0 + self.config.tone_strength * (tone_ratio - 1.0)),
            0.0,
            1.0,
        )

        # ---------------------------------------------------------
        # Step D: Mathematical Color-Field & Palette Projection
        # ---------------------------------------------------------
        # Vectorized soft palette distance & softmax weighting
        # P(p) = sum_k (w_k(p) * c_k)
        flat_pixels = toned_field.reshape(-1, 3)  # (N, 3)
        # Compute squared Euclidean distances to palette colors (N, K)
        # ||p - c_k||^2 = ||p||^2 + ||c_k||^2 - 2 p·c_k
        p_sq = np.sum(flat_pixels**2, axis=1, keepdims=True)  # (N, 1)
        c_sq = np.sum(self.palette**2, axis=1, keepdims=True).T  # (1, K)
        dot = np.dot(flat_pixels, self.palette.T)  # (N, K)
        dists = np.maximum(0.0, p_sq + c_sq - 2.0 * dot)

        tau = 0.04  # Softness temperature
        min_dists = np.min(dists, axis=1, keepdims=True)
        exp_weights = np.exp(-(dists - min_dists) / tau)
        weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

        palette_projected = np.dot(weights, self.palette).reshape(h, w, 3)

        color_field = np.clip(
            (1.0 - self.config.color_palette_mix) * toned_field + self.config.color_palette_mix * palette_projected,
            0.0,
            1.0,
        )

        # Boost saturation in HSV
        if self.config.color_saturation != 1.0:
            hsv = cv2.cvtColor((color_field * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.config.color_saturation, 0.0, 255.0)
            color_field = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        # ---------------------------------------------------------
        # Step E: Cel-Shading Mathematics (Shadows & Highlights)
        # ---------------------------------------------------------
        # Smooth Sigmoidal Shadow field: S = 1 / (1 + exp((Y - Ts) / Ks))
        Ks = 0.06
        shadow_field = 1.0 / (1.0 + np.exp(np.clip((Y - self.config.shadow_threshold) / Ks, -20.0, 20.0)))
        shadow_field = shadow_field[:, :, np.newaxis]

        # Smooth Sigmoidal Highlight field: H = 1 / (1 + exp((Th - Y) / Kh))
        Kh = 0.05
        highlight_field = 1.0 / (1.0 + np.exp(np.clip((self.config.highlight_threshold - Y) / Kh, -20.0, 20.0)))
        highlight_field = highlight_field[:, :, np.newaxis]

        # Apply cel shadows with cool tint shift
        cool_shadow_tint = np.array([-0.03, -0.01, 0.04], dtype=np.float32)
        shaded_color = color_field * (1.0 - self.config.shadow_strength * shadow_field)
        shaded_color = np.clip(shaded_color + shadow_field * cool_shadow_tint * self.config.shadow_strength, 0.0, 1.0)

        # Apply cel highlights
        lit_color = np.clip(shaded_color + self.config.highlight_strength * highlight_field, 0.0, 1.0)

        # ---------------------------------------------------------
        # Step F: Edge Mathematics & Smooth Sigmoidal Line Field
        # ---------------------------------------------------------
        # Gradients: Gx = dY/dx, Gy = dY/dy
        Gx = cv2.Sobel(Y, cv2.CV_32F, 1, 0, ksize=3)
        Gy = cv2.Sobel(Y, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(Gx**2 + Gy**2)

        # Laplacian: L = |∇^2 Y|
        laplacian = np.abs(cv2.Laplacian(Y, cv2.CV_32F, ksize=3))

        # Combined edge field: E = 0.72 G + 0.28 L
        E = 0.72 * grad_mag + 0.28 * laplacian

        # Sigmoid conversion to smooth line field: Es = 1 / (1 + exp(-(E - T) / K))
        Es = 1.0 / (1.0 + np.exp(np.clip(-(E - self.config.edge_threshold) / max(1e-4, self.config.edge_softness), -20.0, 20.0)))
        Es = Es[:, :, np.newaxis]

        # ---------------------------------------------------------
        # Step G: Final Line Field Composition
        # ---------------------------------------------------------
        # Line ink color (Dark charcoal)
        ink_color = np.array([32, 24, 35], dtype=np.float32) / 255.0
        effective_edge = self.config.edge_strength * Es

        # A(x,y) = Ch(x,y) * (1 - Es) + Es * ink_color
        stylized = lit_color * (1.0 - effective_edge) + effective_edge * ink_color
        stylized_uint8 = np.clip(stylized * 255.0, 0.0, 255.0).astype(np.uint8)

        # ---------------------------------------------------------
        # Step H: Vision-Guided Facial & Lip-Sync Modulation
        # ---------------------------------------------------------
        if vision_data is not None:
            # Handle dictionary or FrameVisionData object
            faces = []
            if isinstance(vision_data, dict):
                faces = vision_data.get("faces", [])
            elif hasattr(vision_data, "faces"):
                faces = vision_data.faces or []

            viseme_name = "closed"
            if lipsync_record is not None:
                viseme_name = getattr(lipsync_record, "viseme", "closed") if not isinstance(lipsync_record, dict) else lipsync_record.get("viseme", "closed")

            for f_data in faces:
                from src.art.opencv_renderer import render_anime_mouth
                stylized_uint8 = render_anime_mouth(
                    canvas_rgb=stylized_uint8,
                    face_data=f_data,
                    viseme=viseme_name,
                )

        # ---------------------------------------------------------
        # Step I: Mathematical Temporal Model
        # ---------------------------------------------------------
        if stabilize and self._prev_art is not None and self._prev_luminance is not None:
            # Motion estimation: M_t = 1/N sum |Y_t - Y_{t-1}|
            motion = float(np.mean(np.abs(Y - self._prev_luminance)))

            # If motion is small, lambda -> lambda_max; if motion is large, lambda -> 0
            motion_limit = max(1e-3, self.config.motion_limit)
            lambda_t = self.config.temporal_blend * max(0.0, min(1.0, 1.0 - (motion / motion_limit)))

            # A_t' = (1 - lambda_t) A_t + lambda_t A_{t-1}
            blended = (1.0 - lambda_t) * stylized_uint8.astype(np.float32) + lambda_t * self._prev_art.astype(np.float32)
            stylized_uint8 = np.clip(blended, 0.0, 255.0).astype(np.uint8)

        self._prev_luminance = Y.copy()
        self._prev_art = stylized_uint8.copy()

        return stylized_uint8
