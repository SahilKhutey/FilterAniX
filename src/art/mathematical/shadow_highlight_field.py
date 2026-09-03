from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MathematicalAnimeStyle


@dataclass
class ShadowHighlightFieldResult:
    output_rgb: np.ndarray
    input_rgb: np.ndarray

    luminance: np.ndarray

    small_illumination: np.ndarray
    medium_illumination: np.ndarray
    large_illumination: np.ndarray
    illumination_field: np.ndarray

    shadow_probability: np.ndarray
    highlight_probability: np.ndarray

    shadow_field: np.ndarray
    highlight_field: np.ndarray

    shadow_luminance: np.ndarray
    highlight_luminance: np.ndarray

    target_luminance: np.ndarray

    shadow_rgb: np.ndarray
    highlight_rgb: np.ndarray


class MathematicalShadowHighlightField:
    """
    MTH-06 Mathematical Shadow / Highlight Field Engine.

    Converts luminance into controlled anime-style
    illumination fields.

    The engine constructs:
        multi-scale illumination
        shadow probability
        highlight probability
        shadow modulation
        highlight modulation
        color-temperature modulation

    Input:
        RGB uint8 or floating-point image.

    Output:
        RGB uint8 image.
    """

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:

        self.style = (
            style
            or MathematicalAnimeStyle.creator_anime()
        )

        if hasattr(self.style, "validated"):
            self.style = self.style.validated()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame must be a numpy.ndarray"
            )

        if frame.ndim != 3:
            raise ValueError(
                f"frame must have shape HxWx3, "
                f"got {frame.shape}"
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "frame must have exactly 3 channels"
            )

        if (
            frame.shape[0] < 2
            or frame.shape[1] < 2
        ):
            raise ValueError(
                "frame is too small"
            )

    # =========================================================
    # Normalization
    # =========================================================

    def _normalize(
        self,
        frame: np.ndarray,
    ) -> np.ndarray:

        original_dtype = frame.dtype

        frame = frame.astype(
            np.float32,
            copy=False,
        )

        if np.issubdtype(
            original_dtype,
            np.integer,
        ):
            frame /= 255.0

        elif frame.max() > 1.0:
            frame /= 255.0

        return np.clip(
            frame,
            0.0,
            1.0,
        )

    # =========================================================
    # Luminance
    # =========================================================

    def calculate_luminance(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        return (
            0.2126 * r
            + 0.7152 * g
            + 0.0722 * b
        ).astype(
            np.float32
        )

    # =========================================================
    # Gaussian field
    # =========================================================

    def gaussian_field(
        self,
        luminance: np.ndarray,
        sigma: float,
    ) -> np.ndarray:

        sigma = max(
            float(sigma),
            0.01,
        )

        kernel = max(
            3,
            int(
                round(
                    sigma * 6.0 + 1
                )
            ),
        )

        if kernel % 2 == 0:
            kernel += 1

        return cv2.GaussianBlur(
            luminance,
            (
                kernel,
                kernel,
            ),
            sigmaX=sigma,
            sigmaY=sigma,
        ).astype(
            np.float32
        )

    # =========================================================
    # Multi-scale illumination
    # =========================================================

    def calculate_illumination(
        self,
        luminance: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:

        small = self.gaussian_field(
            luminance,
            self.style.shadow_sigma_small,
        )

        medium = self.gaussian_field(
            luminance,
            self.style.shadow_sigma_medium,
        )

        large = self.gaussian_field(
            luminance,
            self.style.shadow_sigma_large,
        )

        ws = float(
            self.style.shadow_scale_small
        )

        wm = float(
            self.style.shadow_scale_medium
        )

        wl = float(
            self.style.shadow_scale_large
        )

        total = ws + wm + wl

        if total <= 0:
            raise ValueError(
                "Illumination weights must sum to > 0"
            )

        illumination = (
            ws * small
            + wm * medium
            + wl * large
        ) / total

        return (
            small,
            medium,
            large,
            np.clip(
                illumination,
                0.0,
                1.0,
            ).astype(
                np.float32
            ),
        )

    # =========================================================
    # Sigmoid
    # =========================================================

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
            1.0
            / (
                1.0
                + np.exp(-value)
            )
        ).astype(
            np.float32
        )

    # =========================================================
    # Shadow probability
    # =========================================================

    def calculate_shadow_probability(
        self,
        illumination: np.ndarray,
    ) -> np.ndarray:

        threshold = float(
            self.style.shadow_threshold
        )

        softness = max(
            float(
                self.style.shadow_softness
            ),
            1e-5,
        )

        response = (
            threshold
            - illumination
        ) / softness

        return self.sigmoid(
            response
        )

    # =========================================================
    # Highlight probability
    # =========================================================

    def calculate_highlight_probability(
        self,
        illumination: np.ndarray,
    ) -> np.ndarray:

        threshold = float(
            self.style.highlight_threshold
        )

        softness = max(
            float(
                self.style.highlight_softness
            ),
            1e-5,
        )

        response = (
            illumination
            - threshold
        ) / softness

        return self.sigmoid(
            response
        )

    # =========================================================
    # Resolve field interaction
    # =========================================================

    def resolve_interaction(
        self,
        shadow_probability: np.ndarray,
        highlight_probability: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        shadow = (
            shadow_probability
            * (
                1.0
                - highlight_probability
            )
        )

        highlight = (
            highlight_probability
            * (
                1.0
                - shadow_probability
            )
        )

        return (
            np.clip(
                shadow,
                0.0,
                1.0,
            ).astype(np.float32),

            np.clip(
                highlight,
                0.0,
                1.0,
            ).astype(np.float32),
        )

    # =========================================================
    # Target luminance
    # =========================================================

    def calculate_target_luminance(
        self,
        luminance: np.ndarray,
        shadow_field: np.ndarray,
        highlight_field: np.ndarray,
    ) -> np.ndarray:

        shadow_strength = float(
            self.style.shadow_strength
        )

        highlight_strength = float(
            self.style.highlight_strength
        )

        shadow_luminance = (
            luminance
            * (
                1.0
                - shadow_strength
                * shadow_field
            )
        )

        highlight_luminance = (
            shadow_luminance
            + (
                highlight_strength
                * highlight_field
                * (
                    1.0
                    - shadow_luminance
                )
            )
        )

        return np.clip(
            highlight_luminance,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # Shadow color
    # =========================================================

    def apply_shadow_color(
        self,
        rgb: np.ndarray,
        shadow_field: np.ndarray,
    ) -> np.ndarray:

        mix = float(
            self.style.shadow_color_mix
        )

        temperature = float(
            self.style.shadow_temperature
        )

        # Slightly cool anime shadow.
        shadow_tint = np.array(
            [
                0.82,
                0.84,
                1.00,
            ],
            dtype=np.float32,
        )

        tint_strength = (
            mix
            * temperature
            * shadow_field
        )

        result = (
            rgb
            * (
                1.0
                - tint_strength[..., None]
            )
            + (
                rgb
                * shadow_tint
                * tint_strength[..., None]
            )
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # Highlight color
    # =========================================================

    def apply_highlight_color(
        self,
        rgb: np.ndarray,
        highlight_field: np.ndarray,
    ) -> np.ndarray:

        mix = float(
            self.style.highlight_color_mix
        )

        temperature = float(
            self.style.highlight_temperature
        )

        # Warm cinematic highlight.
        highlight_tint = np.array(
            [
                1.00,
                0.94,
                0.82,
            ],
            dtype=np.float32,
        )

        tint_strength = (
            mix
            * temperature
            * highlight_field
        )

        result = (
            rgb
            * (
                1.0
                - tint_strength[..., None]
            )
            + (
                rgb
                * highlight_tint
                * tint_strength[..., None]
            )
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # Luminance reconstruction
    # =========================================================

    def reconstruct_rgb(
        self,
        rgb: np.ndarray,
        original_luminance: np.ndarray,
        target_luminance: np.ndarray,
    ) -> np.ndarray:

        scale = (
            target_luminance
            / (
                original_luminance
                + 1e-6
            )
        )

        scale = np.clip(
            scale,
            0.25,
            1.75,
        )

        result = (
            rgb
            * scale[..., None]
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # Saturation control
    # =========================================================

    def adjust_saturation(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        saturation = float(
            self.style.lighting_saturation
        )

        luminance = self.calculate_luminance(
            rgb
        )

        result = (
            luminance[..., None]
            + (
                rgb
                - luminance[..., None]
            )
            * saturation
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    # =========================================================
    # Full transformation
    # =========================================================

    def transform(
        self,
        frame_rgb: np.ndarray,
    ) -> ShadowHighlightFieldResult:

        self._validate_frame(
            frame_rgb
        )

        rgb = self._normalize(
            frame_rgb
        )

        luminance = self.calculate_luminance(
            rgb
        )

        (
            small,
            medium,
            large,
            illumination,
        ) = self.calculate_illumination(
            luminance
        )

        shadow_probability = (
            self.calculate_shadow_probability(
                illumination
            )
        )

        highlight_probability = (
            self.calculate_highlight_probability(
                illumination
            )
        )

        (
            shadow_field,
            highlight_field,
        ) = self.resolve_interaction(
            shadow_probability,
            highlight_probability,
        )

        target_luminance = (
            self.calculate_target_luminance(
                luminance,
                shadow_field,
                highlight_field,
            )
        )

        output = self.reconstruct_rgb(
            rgb,
            luminance,
            target_luminance,
        )

        shadow_rgb = self.apply_shadow_color(
            output,
            shadow_field,
        )

        highlight_rgb = self.apply_highlight_color(
            shadow_rgb,
            highlight_field,
        )

        output = (
            rgb
            * (
                1.0
                - float(
                    self.style.lighting_global_strength
                )
            )
            + highlight_rgb
            * float(
                self.style.lighting_global_strength
            )
        )

        output = self.adjust_saturation(
            output
        )

        output_rgb = np.clip(
            output * 255.0,
            0.0,
            255.0,
        ).round().astype(
            np.uint8
        )

        return ShadowHighlightFieldResult(
            output_rgb=output_rgb,

            input_rgb=np.clip(
                rgb * 255.0,
                0.0,
                255.0,
            ).round().astype(
                np.uint8
            ),

            luminance=luminance,

            small_illumination=small,
            medium_illumination=medium,
            large_illumination=large,
            illumination_field=illumination,

            shadow_probability=shadow_probability,
            highlight_probability=highlight_probability,

            shadow_field=shadow_field,
            highlight_field=highlight_field,

            shadow_luminance=(
                luminance
                * (
                    1.0
                    - float(
                        self.style.shadow_strength
                    )
                    * shadow_field
                )
            ),

            highlight_luminance=(
                target_luminance
            ),

            target_luminance=target_luminance,

            shadow_rgb=(
                np.clip(
                    shadow_rgb * 255.0,
                    0.0,
                    255.0,
                ).round().astype(
                    np.uint8
                )
            ),

            highlight_rgb=(
                np.clip(
                    highlight_rgb * 255.0,
                    0.0,
                    255.0,
                ).round().astype(
                    np.uint8
                )
            ),
        )

    def render(
        self,
        frame_rgb: np.ndarray,
    ) -> np.ndarray:

        return self.transform(
            frame_rgb
        ).output_rgb
