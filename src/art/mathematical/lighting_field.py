from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .geometry_types import GeometryObservation
from .geometry_field import compute_surface_normals


@dataclass
class LightingFieldResult:
    output_rgb: np.ndarray
    input_rgb: np.ndarray

    luminance: np.ndarray
    local_luminance: np.ndarray
    local_detail: np.ndarray

    shadow_field: np.ndarray
    highlight_field: np.ndarray
    midtone_field: np.ndarray

    local_light_field: np.ndarray
    warm_light_field: np.ndarray

    face_protection_field: np.ndarray
    highlight_protection_field: np.ndarray

    shadow_contribution: np.ndarray
    key_light_contribution: np.ndarray

    final_light_field: np.ndarray


class MathematicalLightingField:
    """
    MTH-09

    Mathematical lighting / illumination field engine.

    This module performs deterministic per-pixel lighting
    transformation.

    It does not generate images and does not use diffusion.
    """

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:

        self.style = (
            style
            or MathematicalAnimeStyle.creator_anime()
        )

        self._validate_style()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_style(self) -> None:

        required = (
            "shadow_threshold",
            "shadow_softness",
            "shadow_strength",
            "highlight_threshold",
            "highlight_softness",
            "highlight_strength",
            "warm_light_strength",
            "warm_light_temperature",
        )

        for name in required:

            if not hasattr(self.style, name):

                raise AttributeError(
                    "MathematicalAnimeStyle is missing "
                    f"required field: {name}"
                )

    def _validate_frame(
        self,
        frame_rgb: np.ndarray,
    ) -> None:

        if not isinstance(
            frame_rgb,
            np.ndarray,
        ):

            raise TypeError(
                "frame_rgb must be numpy.ndarray"
            )

        if frame_rgb.ndim != 3:

            raise ValueError(
                "frame_rgb must have shape (H, W, 3)"
            )

        if frame_rgb.shape[2] != 3:

            raise ValueError(
                "frame_rgb must have exactly 3 channels"
            )

        if (
            frame_rgb.shape[0] < 2
            or frame_rgb.shape[1] < 2
        ):

            raise ValueError(
                "frame_rgb resolution is too small"
            )

    def _validate_observation(
        self,
        observation: GeometryObservation,
        frame_rgb: np.ndarray,
    ) -> None:

        if observation.width != frame_rgb.shape[1]:

            raise ValueError(
                "Geometry observation width does not "
                "match frame width"
            )

        if observation.height != frame_rgb.shape[0]:

            raise ValueError(
                "Geometry observation height does not "
                "match frame height"
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(
        self,
        frame_rgb: np.ndarray,
    ) -> np.ndarray:

        if frame_rgb.dtype == np.uint8:

            return (
                frame_rgb.astype(np.float32)
                / 255.0
            )

        result = frame_rgb.astype(
            np.float32
        )

        if result.max(initial=0.0) > 1.0:

            result /= 255.0

        return np.clip(
            result,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Luminance
    # ------------------------------------------------------------------

    def calculate_luminance(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        return (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Local luminance
    # ------------------------------------------------------------------

    def calculate_local_luminance(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:

        sigma = max(
            float(
                getattr(
                    self.style,
                    "smooth_sigma",
                    1.15,
                )
            ),
            0.0,
        )

        if sigma <= 0.0:

            return luminance.copy()

        result = cv2.GaussianBlur(
            luminance.astype(
                np.float32
            ),
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        return result.astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Local detail
    # ------------------------------------------------------------------

    def calculate_local_detail(
        self,
        luminance: np.ndarray,
        local_luminance: np.ndarray,
    ) -> np.ndarray:

        detail = (
            luminance -
            local_luminance
        )

        return np.clip(
            detail,
            -1.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Sigmoid
    # ------------------------------------------------------------------

    @staticmethod
    def sigmoid(
        value: np.ndarray,
    ) -> np.ndarray:

        value = np.clip(
            value,
            -60.0,
            60.0,
        )

        return (
            1.0 /
            (1.0 + np.exp(-value))
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Shadow
    # ------------------------------------------------------------------

    def calculate_shadow_field(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:

        threshold = float(
            np.clip(
                self.style.shadow_threshold,
                0.0,
                1.0,
            )
        )

        softness = max(
            float(
                self.style.shadow_softness
            ),
            1e-4,
        )

        field = self.sigmoid(
            (
                threshold -
                luminance
            ) /
            softness
        )

        return np.clip(
            field,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Highlight
    # ------------------------------------------------------------------

    def calculate_highlight_field(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:

        threshold = float(
            np.clip(
                self.style.highlight_threshold,
                0.0,
                1.0,
            )
        )

        softness = max(
            float(
                self.style.highlight_softness
            ),
            1e-4,
        )

        field = self.sigmoid(
            (
                luminance -
                threshold
            ) /
            softness
        )

        return np.clip(
            field,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Midtones
    # ------------------------------------------------------------------

    def calculate_midtone_field(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:

        result = (
            1.0 -
            np.abs(
                2.0 * luminance -
                1.0
            )
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Local light
    # ------------------------------------------------------------------

    def calculate_local_light_field(
        self,
        luminance: np.ndarray,
        local_detail: np.ndarray,
    ) -> np.ndarray:

        positive_detail = np.maximum(
            local_detail,
            0.0,
        )

        result = (
            0.65 * luminance +
            0.35 * positive_detail
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Face protection
    # ------------------------------------------------------------------

    def calculate_face_protection(
        self,
        observation: GeometryObservation,
        width: int,
        height: int,
    ) -> np.ndarray:

        if observation.face_box is None:

            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        x0 = float(
            np.clip(
                observation.face_box.x0,
                0,
                width - 1,
            )
        )

        y0 = float(
            np.clip(
                observation.face_box.y0,
                0,
                height - 1,
            )
        )

        x1 = float(
            np.clip(
                observation.face_box.x1,
                0,
                width - 1,
            )
        )

        y1 = float(
            np.clip(
                observation.face_box.y1,
                0,
                height - 1,
            )
        )

        if x1 < x0:
            x0, x1 = x1, x0

        if y1 < y0:
            y0, y1 = y1, y0

        y, x = np.mgrid[
            0:height,
            0:width,
        ]

        dx = np.maximum(
            np.maximum(
                x0 - x,
                0.0,
            ),
            x - x1,
        )

        dy = np.maximum(
            np.maximum(
                y0 - y,
                0.0,
            ),
            y - y1,
        )

        distance_sq = (
            dx * dx +
            dy * dy
        )

        sigma = max(
            1.0,
            min(width, height) * 0.03,
        )

        field = np.exp(
            -distance_sq /
            (2.0 * sigma * sigma)
        )

        confidence = float(
            np.clip(
                getattr(
                    observation.face_box,
                    "confidence",
                    1.0,
                ),
                0.0,
                1.0,
            )
        )

        return np.clip(
            field * confidence,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Key light
    # ------------------------------------------------------------------

    def get_key_light(
        self,
    ) -> np.ndarray:

        key_light = np.asarray(
            getattr(
                self.style,
                "key_light",
                getattr(self.style, "key_light_color", (255, 239, 211)),
            ),
            dtype=np.float32,
        )

        if key_light.max(initial=0.0) > 1.0:

            key_light /= 255.0

        return np.clip(
            key_light,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Shadow color
    # ------------------------------------------------------------------

    def get_shadow_color(
        self,
    ) -> np.ndarray:

        shadow_color = np.asarray(
            getattr(
                self.style,
                "shadow_color",
                (66, 70, 94),
            ),
            dtype=np.float32,
        )

        if shadow_color.max(initial=0.0) > 1.0:

            shadow_color /= 255.0

        return np.clip(
            shadow_color,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Warm light
    # ------------------------------------------------------------------

    def calculate_warm_light_field(
        self,
        local_light_field: np.ndarray,
        highlight_field: np.ndarray,
        face_protection: np.ndarray,
    ) -> np.ndarray:

        strength = float(
            np.clip(
                self.style.warm_light_strength,
                0.0,
                1.0,
            )
        )

        temperature = float(
            np.clip(
                self.style.warm_light_temperature,
                0.0,
                1.0,
            )
        )

        result = (
            local_light_field *
            (
                0.70 +
                0.30 * highlight_field
            )
        )

        result *= (
            1.0 +
            0.15 *
            temperature *
            face_protection
        )

        result *= strength

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature-aware highlight protection
    # ------------------------------------------------------------------

    def calculate_highlight_protection(
        self,
        highlight_field: np.ndarray,
        face_protection: np.ndarray,
    ) -> np.ndarray:

        result = (
            0.75 * highlight_field +
            0.25 * face_protection
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Apply shadow
    # ------------------------------------------------------------------

    def apply_shadow(
        self,
        rgb: np.ndarray,
        shadow_field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        strength = float(
            np.clip(
                self.style.shadow_strength,
                0.0,
                1.0,
            )
        )

        shadow_color = (
            self.get_shadow_color()
        )

        contribution = (
            shadow_field *
            strength
        )

        output = (
            rgb *
            (
                1.0 -
                contribution[..., None]
            )
            +
            shadow_color[
                None,
                None,
                :
            ] *
            contribution[..., None]
        )

        return (
            np.clip(
                output,
                0.0,
                1.0,
            ),
            contribution.astype(
                np.float32
            ),
        )

    # ------------------------------------------------------------------
    # Apply key light
    # ------------------------------------------------------------------

    def apply_key_light(
        self,
        rgb: np.ndarray,
        warm_light_field: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        key_light = self.get_key_light()

        temperature = float(
            np.clip(
                self.style.warm_light_temperature,
                0.0,
                1.0,
            )
        )

        # Move configured key light toward
        # warm cinematic illumination.
        warm_target = np.array(
            [1.0, 0.93, 0.82],
            dtype=np.float32,
        )

        key_light = (
            (1.0 - temperature) *
            key_light
            +
            temperature *
            warm_target
        )

        contribution = (
            warm_light_field
        )

        output = (
            rgb *
            (
                1.0 -
                contribution[..., None]
            )
            +
            key_light[
                None,
                None,
                :
            ] *
            contribution[..., None]
        )

        return (
            np.clip(
                output,
                0.0,
                1.0,
            ),
            contribution.astype(
                np.float32
            ),
        )

    # ------------------------------------------------------------------
    # Highlights
    # ------------------------------------------------------------------

    def apply_highlights(
        self,
        rgb: np.ndarray,
        highlight_protection: np.ndarray,
    ) -> np.ndarray:

        strength = float(
            np.clip(
                self.style.highlight_strength,
                0.0,
                1.0,
            )
        )

        key_light = self.get_key_light()

        highlight_amount = (
            highlight_protection *
            strength
        )

        output = (
            rgb +
            key_light[
                None,
                None,
                :
            ] *
            highlight_amount[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Complete transform
    # ------------------------------------------------------------------

    def transform(
        self,
        frame_rgb: np.ndarray,
        observation: GeometryObservation,
    ) -> LightingFieldResult:

        self._validate_frame(
            frame_rgb
        )

        self._validate_observation(
            observation,
            frame_rgb,
        )

        rgb = self._normalize(
            frame_rgb
        )

        height, width = (
            rgb.shape[:2]
        )

        luminance = (
            self.calculate_luminance(
                rgb
            )
        )

        local_luminance = (
            self.calculate_local_luminance(
                luminance
            )
        )

        local_detail = (
            self.calculate_local_detail(
                luminance,
                local_luminance,
            )
        )

        shadow_field = (
            self.calculate_shadow_field(
                luminance
            )
        )

        highlight_field = (
            self.calculate_highlight_field(
                luminance
            )
        )

        midtone_field = (
            self.calculate_midtone_field(
                luminance
            )
        )

        local_light_field = (
            self.calculate_local_light_field(
                luminance,
                local_detail,
            )
        )

        face_protection = (
            self.calculate_face_protection(
                observation,
                width,
                height,
            )
        )

        warm_light_field = (
            self.calculate_warm_light_field(
                local_light_field,
                highlight_field,
                face_protection,
            )
        )

        highlight_protection = (
            self.calculate_highlight_protection(
                highlight_field,
                face_protection,
            )
        )

        shadow_rgb, shadow_contribution = (
            self.apply_shadow(
                rgb,
                shadow_field,
            )
        )

        light_rgb, key_light_contribution = (
            self.apply_key_light(
                shadow_rgb,
                warm_light_field,
            )
        )

        output_rgb = (
            self.apply_highlights(
                light_rgb,
                highlight_protection,
            )
        )

        final_light_field = np.clip(
            (
                0.45 * local_light_field
                +
                0.25 * highlight_field
                +
                0.30 * face_protection
            ),
            0.0,
            1.0,
        ).astype(np.float32)

        output_uint8 = np.clip(
            output_rgb * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8)

        return LightingFieldResult(
            output_rgb=output_uint8,
            input_rgb=frame_rgb.copy(),

            luminance=luminance,
            local_luminance=local_luminance,
            local_detail=local_detail,

            shadow_field=shadow_field,
            highlight_field=highlight_field,
            midtone_field=midtone_field,

            local_light_field=local_light_field,
            warm_light_field=warm_light_field,

            face_protection_field=face_protection,
            highlight_protection_field=highlight_protection,

            shadow_contribution=shadow_contribution,
            key_light_contribution=key_light_contribution,

            final_light_field=final_light_field,
        )

    def render(
        self,
        frame_rgb: np.ndarray,
        observation: GeometryObservation,
    ) -> np.ndarray:

        return self.transform(
            frame_rgb,
            observation,
        ).output_rgb


# ======================================================================
# Compatibility Function for Existing Pipelines
# ======================================================================

def compute_lighting_field(
    art_field: np.ndarray,
    luminance: np.ndarray,
    style: MathematicalAnimeStyle,
    light_dir: Tuple[float, float, float] = (-0.35, -0.45, 0.82),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compatibility function: Computes directional cinematic key light.
    """
    if style.warm_light_strength <= 0.0:
        return art_field, np.zeros((art_field.shape[0], art_field.shape[1], 1), dtype=np.float32)

    normals = compute_surface_normals(luminance)
    l_vec = np.array(light_dir, dtype=np.float32)
    l_norm = np.linalg.norm(l_vec)
    if l_norm > 1e-6:
        l_vec = l_vec / l_norm

    dot = np.sum(normals * l_vec, axis=-1, keepdims=True)
    key_light = np.maximum(0.0, dot)

    alpha_l = style.warm_light_strength
    warm_k = np.array(getattr(style, "key_light_color", (255, 239, 211)), dtype=np.float32)
    if warm_k.max() > 1.0:
        warm_k /= 255.0

    lit = np.clip(art_field + alpha_l * key_light * warm_k, 0.0, 1.0)
    return lit, key_light
