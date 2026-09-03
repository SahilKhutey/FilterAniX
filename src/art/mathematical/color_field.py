from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MathematicalAnimeStyle


@dataclass(frozen=True)
class ColorFieldResult:
    """
    Intermediate mathematical fields produced by MTH-02.
    """

    output_rgb: np.ndarray

    smoothed_rgb: np.ndarray

    luminance: np.ndarray

    quantized_luminance: np.ndarray

    saturation: np.ndarray

    palette_rgb: np.ndarray


class MathematicalColorField:
    """
    MTH-02 Mathematical Color Field.

    Converts every input pixel into a new mathematically
    calculated color.

    CPU-oriented implementation.

    Input:
        RGB uint8 image, shape (H, W, 3)

    Output:
        RGB uint8 image, shape (H, W, 3)
    """

    def __init__(
        self,
        style: Optional[MathematicalAnimeStyle] = None,
    ) -> None:

        self.style = (
            style or MathematicalAnimeStyle.creator_anime()
        ).validated()

        self._palette = (
            np.asarray(
                self.style.palette,
                dtype=np.float32,
            )
            / 255.0
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_frame(
        rgb: np.ndarray,
    ) -> None:

        if not isinstance(rgb, np.ndarray):
            raise TypeError(
                "Input frame must be a numpy.ndarray"
            )

        if rgb.ndim != 3:
            raise ValueError(
                "Input frame must have shape (H, W, 3)"
            )

        if rgb.shape[2] != 3:
            raise ValueError(
                "Input frame must contain 3 RGB channels"
            )

        if rgb.shape[0] < 1 or rgb.shape[1] < 1:
            raise ValueError(
                "Input frame cannot be empty"
            )

    # ============================================================
    # NORMALIZATION
    # ============================================================

    @staticmethod
    def _normalize(
        rgb: np.ndarray,
    ) -> np.ndarray:

        if rgb.dtype == np.uint8:

            return (
                rgb.astype(
                    np.float32
                )
                / 255.0
            )

        if np.issubdtype(
            rgb.dtype,
            np.floating,
        ):

            result = rgb.astype(
                np.float32,
                copy=False,
            )

            if (
                np.nanmin(result) >= 0.0
                and np.nanmax(result) <= 1.0
            ):
                return result

            return np.clip(
                result / 255.0,
                0.0,
                1.0,
            )

        raise TypeError(
            "RGB frame must use uint8 or floating-point data"
        )

    # ============================================================
    # LUMINANCE
    # ============================================================

    @staticmethod
    def luminance(
        rgb: np.ndarray,
    ) -> np.ndarray:

        return (
            0.2126 * rgb[..., 0]
            + 0.7152 * rgb[..., 1]
            + 0.0722 * rgb[..., 2]
        )

    # ============================================================
    # LOCAL COLOR FIELD
    # ============================================================

    def smooth_color_field(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        sigma = self.style.smooth_sigma

        if sigma <= 0.0:
            return rgb.copy()

        # Convert sigma to a compact bilateral neighborhood.
        diameter = max(
            3,
            int(round(sigma * 6.0)) | 1,
        )

        result = cv2.bilateralFilter(
            rgb.astype(np.float32),
            diameter,
            sigmaColor=0.10,
            sigmaSpace=max(1.0, sigma * 4.0),
        )

        return np.clip(
            result,
            0.0,
            1.0,
        )

    # ============================================================
    # TONE QUANTIZATION
    # ============================================================

    def quantize_luminance(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:

        levels = max(
            2,
            int(self.style.color_levels),
        )

        scaled = (
            luminance
            * float(levels - 1)
        )

        quantized = (
            np.round(scaled)
            / float(levels - 1)
        )

        return np.clip(
            quantized,
            0.0,
            1.0,
        )

    # ============================================================
    # SATURATION
    # ============================================================

    @staticmethod
    def _rgb_to_hsv(
        rgb: np.ndarray,
    ) -> np.ndarray:

        rgb8 = np.clip(
            rgb * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        hsv8 = cv2.cvtColor(
            rgb8,
            cv2.COLOR_RGB2HSV,
        )

        return (
            hsv8.astype(np.float32)
            / np.array(
                [179.0, 255.0, 255.0],
                dtype=np.float32,
            )
        )

    @staticmethod
    def _hsv_to_rgb(
        hsv: np.ndarray,
    ) -> np.ndarray:

        hsv8 = np.empty_like(
            hsv,
            dtype=np.uint8,
        )

        hsv8[..., 0] = np.clip(
            hsv[..., 0] * 179.0,
            0.0,
            179.0,
        ).astype(np.uint8)

        hsv8[..., 1] = np.clip(
            hsv[..., 1] * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        hsv8[..., 2] = np.clip(
            hsv[..., 2] * 255.0,
            0.0,
            255.0,
        ).astype(np.uint8)

        rgb8 = cv2.cvtColor(
            hsv8,
            cv2.COLOR_HSV2RGB,
        )

        return (
            rgb8.astype(np.float32)
            / 255.0
        )

    def adjust_saturation(
        self,
        rgb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        hsv = self._rgb_to_hsv(rgb)

        saturation = hsv[..., 1].copy()

        multiplier = float(
            self.style.saturation
        )

        saturation_curve = np.power(
            np.clip(
                saturation,
                0.0,
                1.0,
            ),
            0.92,
        )

        saturation_new = np.clip(
            saturation_curve * multiplier,
            0.0,
            1.0,
        )

        hsv[..., 1] = saturation_new

        output = self._hsv_to_rgb(hsv)

        return output, saturation_new

    # ============================================================
    # TONE FIELD
    # ============================================================

    def apply_tone_field(
        self,
        rgb: np.ndarray,
        luminance: np.ndarray,
        quantized_luminance: np.ndarray,
    ) -> np.ndarray:

        # Avoid division instability in dark pixels.
        safe_luminance = np.maximum(
            luminance,
            1e-4,
        )

        ratio = (
            quantized_luminance
            / safe_luminance
        )

        ratio = np.clip(
            ratio,
            0.50,
            1.80,
        )

        strength = float(
            self.style.tone_strength
        )

        scale = (
            1.0 - strength
        ) + (
            strength * ratio
        )

        output = (
            rgb
            * scale[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        )

    # ============================================================
    # PALETTE PROJECTION
    # ============================================================

    def project_palette(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        """
        Projects each pixel onto the configured palette.

        We retain the nearest and second-nearest palette colors
        rather than constructing a huge HxWxK distance tensor.

        This substantially reduces memory usage.
        """

        height, width = rgb.shape[:2]

        nearest_distance = np.full(
            (height, width),
            np.inf,
            dtype=np.float32,
        )

        second_distance = np.full(
            (height, width),
            np.inf,
            dtype=np.float32,
        )

        nearest_index = np.zeros(
            (height, width),
            dtype=np.int32,
        )

        second_index = np.zeros(
            (height, width),
            dtype=np.int32,
        )

        for index, color in enumerate(
            self._palette
        ):

            diff = (
                rgb
                - color[None, None, :]
            )

            distance = np.sum(
                diff * diff,
                axis=2,
            )

            better = (
                distance
                < nearest_distance
            )

            second_better = (
                (~better)
                & (
                    distance
                    < second_distance
                )
            )

            second_distance = np.where(
                better,
                nearest_distance,
                np.where(
                    second_better,
                    distance,
                    second_distance,
                ),
            )

            second_index = np.where(
                better,
                nearest_index,
                np.where(
                    second_better,
                    index,
                    second_index,
                ),
            )

            nearest_distance = np.where(
                better,
                distance,
                nearest_distance,
            )

            nearest_index = np.where(
                better,
                index,
                nearest_index,
            )

        nearest_color = self._palette[
            nearest_index
        ]

        second_color = self._palette[
            second_index
        ]

        total = (
            nearest_distance
            + second_distance
            + 1e-8
        )

        # Distance-based interpolation.
        #
        # If second color is much farther away,
        # alpha approaches zero.
        alpha = (
            nearest_distance
            / total
        )

        alpha = np.clip(
            alpha,
            0.0,
            0.5,
        ) * 2.0

        projected = (
            nearest_color
            * (1.0 - alpha[..., None])
            + second_color
            * alpha[..., None]
        )

        return np.clip(
            projected,
            0.0,
            1.0,
        )

    # ============================================================
    # COLOR FIELD COMPOSITION
    # ============================================================

    def compose_color_field(
        self,
        original: np.ndarray,
        palette_rgb: np.ndarray,
    ) -> np.ndarray:

        mix = float(
            self.style.palette_mix
        )

        result = (
            original
            * (1.0 - mix)
            + palette_rgb
            * mix
        )

        return np.clip(
            result,
            0.0,
            1.0,
        )

    # ============================================================
    # MAIN TRANSFORMATION
    # ============================================================

    def transform(
        self,
        rgb: np.ndarray,
    ) -> ColorFieldResult:

        self._validate_frame(rgb)

        normalized = self._normalize(rgb)

        # --------------------------------------------------------
        # 1. Smooth photographic micro-texture
        # --------------------------------------------------------

        smoothed = self.smooth_color_field(
            normalized
        )

        # --------------------------------------------------------
        # 2. Luminance field
        # --------------------------------------------------------

        luminance = self.luminance(
            smoothed
        )

        # --------------------------------------------------------
        # 3. Tone quantization
        # --------------------------------------------------------

        quantized = self.quantize_luminance(
            luminance
        )

        # --------------------------------------------------------
        # 4. Apply mathematical tone field
        # --------------------------------------------------------

        tone_field = self.apply_tone_field(
            smoothed,
            luminance,
            quantized,
        )

        # --------------------------------------------------------
        # 5. Saturation transformation
        # --------------------------------------------------------

        saturated, saturation = (
            self.adjust_saturation(
                tone_field
            )
        )

        # --------------------------------------------------------
        # 6. Palette projection
        # --------------------------------------------------------

        palette_rgb = self.project_palette(
            saturated
        )

        # --------------------------------------------------------
        # 7. Original field ↔ palette field
        # --------------------------------------------------------

        output = self.compose_color_field(
            saturated,
            palette_rgb,
        )

        # --------------------------------------------------------
        # 8. Global gamma
        # --------------------------------------------------------

        gamma = float(
            self.style.gamma
        )

        output = np.power(
            np.clip(
                output,
                0.0,
                1.0,
            ),
            gamma,
        )

        # --------------------------------------------------------
        # 9. Contrast
        # --------------------------------------------------------

        contrast = float(
            self.style.contrast
        )

        output = (
            (
                output - 0.5
            )
            * contrast
            + 0.5
        )

        output = np.clip(
            output,
            0.0,
            1.0,
        )

        # --------------------------------------------------------
        # 10. Final uint8 conversion
        # --------------------------------------------------------

        output_rgb = (
            output * 255.0
        ).round().astype(
            np.uint8
        )

        smoothed_rgb = (
            smoothed * 255.0
        ).round().astype(
            np.uint8
        )

        palette_uint8 = (
            palette_rgb * 255.0
        ).round().astype(
            np.uint8
        )

        return ColorFieldResult(
            output_rgb=output_rgb,
            smoothed_rgb=smoothed_rgb,
            luminance=luminance,
            quantized_luminance=quantized,
            saturation=saturation,
            palette_rgb=palette_uint8,
        )

    # ============================================================
    # SIMPLE API
    # ============================================================

    def render(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        return self.transform(
            rgb
        ).output_rgb


def compute_color_field(
    rgb_uint8: np.ndarray,
    style: Optional[MathematicalAnimeStyle] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function preserving backwards compatibility with pipeline compositor.
    Returns (color_field_float, smoothed_base_float) in [0.0, 1.0] float32 RGB.
    """
    engine = MathematicalColorField(style)
    res = engine.transform(rgb_uint8)
    return (
        res.output_rgb.astype(np.float32) / 255.0,
        res.smoothed_rgb.astype(np.float32) / 255.0,
    )
