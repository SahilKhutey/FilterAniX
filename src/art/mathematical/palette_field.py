from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle
from .tone_field import ToneFieldResult


@dataclass(frozen=True)
class PaletteFieldResult:
    """
    Complete mathematical palette fields produced by MTH-04.
    """

    output_rgb: np.ndarray

    input_rgb: np.ndarray

    palette_rgb: np.ndarray

    palette_lab: np.ndarray

    weights: np.ndarray

    dominant_index: np.ndarray

    confidence: np.ndarray

    palette_entropy: np.ndarray

    luminance_before: np.ndarray

    luminance_after: np.ndarray


class MathematicalPaletteField:
    """
    MTH-04 Mathematical Palette Field Engine.

    Converts the MTH-03 tone/color result into a controlled
    anime-oriented palette field.

    Every pixel receives a mathematically calculated palette
    contribution.

    Input:
        RGB uint8 [0,255]
        or floating RGB [0,1]

    Output:
        RGB uint8 [0,255]
    """

    def __init__(
        self,
        style: Optional[MathematicalAnimeStyle] = None,
    ) -> None:

        self.style = (
            style
            or MathematicalAnimeStyle.creator_anime()
        ).validated()

        self._palette_rgb = np.asarray(
            self.style.palette,
            dtype=np.float32,
        )

        if (
            self._palette_rgb.ndim != 2
            or self._palette_rgb.shape[1] != 3
        ):
            raise ValueError(
                "Palette must have shape (N, 3)"
            )

        self._palette_rgb_normalized = (
            self._palette_rgb / 255.0
        )

        self._palette_lab = (
            self._rgb_to_lab(
                self._palette_rgb_normalized
            )
        )

    # ============================================================
    # VALIDATION
    # ============================================================

    @staticmethod
    def _validate_frame(
        rgb: np.ndarray,
    ) -> None:

        if not isinstance(
            rgb,
            np.ndarray,
        ):
            raise TypeError(
                "Input frame must be numpy.ndarray"
            )

        if rgb.ndim != 3:
            raise ValueError(
                "Input frame must have shape (H,W,3)"
            )

        if rgb.shape[2] != 3:
            raise ValueError(
                "Input frame must have 3 channels"
            )

        if rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
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

            value = rgb.astype(
                np.float32,
                copy=False,
            )

            maximum = float(
                np.nanmax(value)
            )

            minimum = float(
                np.nanmin(value)
            )

            if (
                minimum >= 0.0
                and maximum <= 1.0
            ):
                return np.clip(
                    value,
                    0.0,
                    1.0,
                )

            return np.clip(
                value / 255.0,
                0.0,
                1.0,
            )

        raise TypeError(
            "Input must be uint8 or floating point"
        )

    # ============================================================
    # RGB → LAB
    # ============================================================

    @staticmethod
    def _rgb_to_lab(
        rgb: np.ndarray,
    ) -> np.ndarray:

        rgb = np.asarray(
            rgb,
            dtype=np.float32,
        )

        if rgb.ndim == 2:

            rgb = rgb.reshape(
                1,
                -1,
                3,
            )

            lab = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2LAB,
            )

            return lab.reshape(
                -1,
                3,
            )

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2LAB,
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
    # PALETTE DISTANCE
    # ============================================================

    def calculate_distances(
        self,
        lab: np.ndarray,
    ) -> np.ndarray:

        """
        Calculate squared Lab distance from every pixel
        to every palette color.

        Output:
            H x W x K
        """

        difference = (
            lab[..., None, :]
            - self._palette_lab[None, None, :, :]
        )

        distances = np.sum(
            difference * difference,
            axis=-1,
        )

        return distances

    # ============================================================
    # SOFT PALETTE WEIGHTS
    # ============================================================

    def calculate_weights(
        self,
        distances: np.ndarray,
    ) -> np.ndarray:

        """
        Convert distances to differentiable-like soft
        palette assignments.

        Lower distance = larger weight.
        """

        temperature = (
            8.0
            + (
                1.0
                - float(
                    self.style.palette_temperature
                )
            )
            * 24.0
        )

        temperature = max(
            1e-3,
            temperature,
        )

        logits = (
            -distances
            / temperature
        )

        logits = (
            logits
            - np.max(
                logits,
                axis=-1,
                keepdims=True,
            )
        )

        weights = np.exp(
            logits
        )

        denominator = np.maximum(
            np.sum(
                weights,
                axis=-1,
                keepdims=True,
            ),
            1e-8,
        )

        weights = (
            weights
            / denominator
        )

        return np.clip(
            weights,
            0.0,
            1.0,
        )

    # ============================================================
    # PALETTE FIELD
    # ============================================================

    def calculate_palette_field(
        self,
        weights: np.ndarray,
    ) -> np.ndarray:

        """
        Weighted combination of all palette colors.
        """

        palette = np.sum(
            weights[..., None]
            * self._palette_rgb_normalized[
                None,
                None,
                :,
                :,
            ],
            axis=-2,
        )

        return np.clip(
            palette,
            0.0,
            1.0,
        )

    # ============================================================
    # DOMINANT PALETTE
    # ============================================================

    @staticmethod
    def dominant_palette(
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        index = np.argmax(
            weights,
            axis=-1,
        )

        confidence = np.max(
            weights,
            axis=-1,
        )

        return (
            index,
            confidence,
        )

    # ============================================================
    # PALETTE ENTROPY
    # ============================================================

    @staticmethod
    def calculate_entropy(
        weights: np.ndarray,
    ) -> np.ndarray:

        safe = np.clip(
            weights,
            1e-8,
            1.0,
        )

        entropy = -np.sum(
            safe * np.log(
                safe
            ),
            axis=-1,
        )

        palette_count = (
            weights.shape[-1]
        )

        if palette_count > 1:

            entropy /= np.log(
                palette_count
            )

        return np.clip(
            entropy,
            0.0,
            1.0,
        )

    # ============================================================
    # PALETTE BLEND
    # ============================================================

    def blend_palette(
        self,
        source_rgb: np.ndarray,
        palette_rgb: np.ndarray,
    ) -> np.ndarray:

        mix = float(
            self.style.palette_mix
        )

        mix = float(
            np.clip(
                mix,
                0.0,
                1.0,
            )
        )

        result = (
            source_rgb
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
    # LUMINANCE PRESERVATION
    # ============================================================

    def restore_luminance(
        self,
        rgb: np.ndarray,
        target_luminance: np.ndarray,
    ) -> np.ndarray:

        current_luminance = (
            self.luminance(
                rgb
            )
        )

        ratio = (
            target_luminance
            / np.maximum(
                current_luminance,
                1e-4,
            )
        )

        # Prevent extreme amplification in very dark pixels.
        ratio = np.clip(
            ratio,
            0.20,
            3.0,
        )

        result = (
            rgb
            * ratio[..., None]
        )

        return np.clip(
            result,
            0.0,
            1.0,
        )

    # ============================================================
    # WARM LIGHT FIELD
    # ============================================================

    def apply_warm_light(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:

        strength = float(
            self.style.warm_light_strength
        )

        temperature = float(
            self.style.warm_light_temperature
        )

        strength = np.clip(
            strength,
            0.0,
            1.0,
        )

        temperature = np.clip(
            temperature,
            0.0,
            1.0,
        )

        key_light = np.array(
            [
                1.0,
                0.93,
                0.82,
            ],
            dtype=np.float32,
        )

        warm_factor = (
            strength
            * temperature
        )

        result = (
            rgb
            * (
                1.0
                - warm_factor
            )
            + key_light
            * warm_factor
            * rgb
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
        tone_result: Optional[
            ToneFieldResult
        ] = None,
    ) -> PaletteFieldResult:

        self._validate_frame(
            rgb
        )

        source = self._normalize(
            rgb
        )

        # --------------------------------------------------------
        # MTH-03 target luminance
        # --------------------------------------------------------

        if tone_result is not None:

            target_luminance = (
                tone_result.target_luminance
            )

        else:

            target_luminance = (
                self.luminance(
                    source
                )
            )

        # --------------------------------------------------------
        # RGB → Lab
        # --------------------------------------------------------

        lab = self._rgb_to_lab(
            source
        )

        # --------------------------------------------------------
        # Distance to palette
        # --------------------------------------------------------

        distances = (
            self.calculate_distances(
                lab
            )
        )

        # --------------------------------------------------------
        # Soft palette weights
        # --------------------------------------------------------

        weights = (
            self.calculate_weights(
                distances
            )
        )

        # --------------------------------------------------------
        # Continuous palette field
        # --------------------------------------------------------

        palette_rgb = (
            self.calculate_palette_field(
                weights
            )
        )

        # --------------------------------------------------------
        # Blend source and palette
        # --------------------------------------------------------

        blended = (
            self.blend_palette(
                source,
                palette_rgb,
            )
        )

        # --------------------------------------------------------
        # Restore MTH-03 luminance structure
        # --------------------------------------------------------

        tone_preserved = (
            self.restore_luminance(
                blended,
                target_luminance,
            )
        )

        # --------------------------------------------------------
        # Warm cinematic field
        # --------------------------------------------------------

        output = (
            self.apply_warm_light(
                tone_preserved
            )
        )

        # --------------------------------------------------------
        # Dominant palette information
        # --------------------------------------------------------

        dominant_index, confidence = (
            self.dominant_palette(
                weights
            )
        )

        # --------------------------------------------------------
        # Palette entropy
        # --------------------------------------------------------

        entropy = (
            self.calculate_entropy(
                weights
            )
        )

        # --------------------------------------------------------
        # Final uint8 output
        # --------------------------------------------------------

        output_rgb = (
            output
            * 255.0
        ).round().astype(
            np.uint8
        )

        input_rgb = (
            source
            * 255.0
        ).round().astype(
            np.uint8
        )

        palette_rgb_uint8 = (
            palette_rgb
            * 255.0
        ).round().astype(
            np.uint8
        )

        luminance_before = (
            self.luminance(
                source
            )
        )

        luminance_after = (
            self.luminance(
                output
            )
        )

        return PaletteFieldResult(
            output_rgb=output_rgb,
            input_rgb=input_rgb,
            palette_rgb=palette_rgb_uint8,
            palette_lab=self._palette_lab.copy(),
            weights=weights,
            dominant_index=dominant_index,
            confidence=confidence,
            palette_entropy=entropy,
            luminance_before=luminance_before,
            luminance_after=luminance_after,
        )

    # ============================================================
    # SIMPLE API
    # ============================================================

    def render(
        self,
        rgb: np.ndarray,
        tone_result: Optional[
            ToneFieldResult
        ] = None,
    ) -> np.ndarray:

        return self.transform(
            rgb,
            tone_result,
        ).output_rgb


def compute_palette_projection(
    toned_field: np.ndarray,
    style: MathematicalAnimeStyle,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compatibility function for pipeline compositor.
    Returns projected_field in [0.0, 1.0] float32 RGB.
    """
    engine = MathematicalPaletteField(style)
    res = engine.transform(toned_field)
    return res.output_rgb.astype(np.float32) / 255.0
