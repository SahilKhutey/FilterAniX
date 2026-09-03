from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MathematicalAnimeStyle


@dataclass
class EdgeFieldResult:
    output_rgb: np.ndarray
    input_rgb: np.ndarray

    luminance: np.ndarray

    gradient_x: np.ndarray
    gradient_y: np.ndarray
    gradient_magnitude: np.ndarray
    gradient_orientation: np.ndarray

    laplacian: np.ndarray

    small_scale_response: np.ndarray
    medium_scale_response: np.ndarray
    large_scale_response: np.ndarray

    multiscale_response: np.ndarray

    edge_probability: np.ndarray
    line_strength: np.ndarray
    line_field: np.ndarray


class MathematicalEdgeField:
    """
    MTH-05 Mathematical Edge / Line Field Engine.

    Converts image structure into a continuous mathematical line field.

    Input:
        RGB uint8 or floating point image.

    Output:
        RGB uint8 image.

    The engine does not use Canny edge detection.
    It constructs a continuous field from:
        gradient magnitude
        Laplacian response
        multi-scale structure
        soft thresholding
    """

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:
        self.style = style or MathematicalAnimeStyle.creator_anime()

        if hasattr(self.style, "validated"):
            self.style = self.style.validated()

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim != 3:
            raise ValueError(
                f"frame must have shape HxWx3, got {frame.shape}"
            )

        if frame.shape[2] != 3:
            raise ValueError(
                f"frame must have 3 channels, got {frame.shape[2]}"
            )

        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError("frame is too small")

    # ---------------------------------------------------------
    # Normalization (Section 5 safe version)
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Luminance
    # ---------------------------------------------------------

    def calculate_luminance(
        self,
        rgb: np.ndarray,
    ) -> np.ndarray:
        """
        ITU-R BT.709 luminance.
        """

        r = rgb[..., 0]
        g = rgb[..., 1]
        b = rgb[..., 2]

        return (
            0.2126 * r
            + 0.7152 * g
            + 0.0722 * b
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Gradient
    # ---------------------------------------------------------

    def calculate_gradient(
        self,
        luminance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate spatial gradient.

        Returns:
            gradient_x
            gradient_y
            magnitude
            orientation
        """

        gx = cv2.Sobel(
            luminance,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        gy = cv2.Sobel(
            luminance,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        magnitude = np.sqrt(
            gx * gx + gy * gy
        )

        orientation = np.arctan2(gy, gx)

        return (
            gx.astype(np.float32),
            gy.astype(np.float32),
            magnitude.astype(np.float32),
            orientation.astype(np.float32),
        )

    # ---------------------------------------------------------
    # Normalize field
    # ---------------------------------------------------------

    def normalize_field(
        self,
        field: np.ndarray,
        percentile: float | None = None,
    ) -> np.ndarray:
        """
        Robustly normalize a field using percentile scaling.
        """

        if percentile is None:
            percentile = self.style.edge_percentile

        positive = np.abs(field)

        scale = float(
            np.percentile(
                positive,
                percentile,
            )
        )

        if scale < 1e-8:
            return np.zeros_like(
                field,
                dtype=np.float32,
            )

        return np.clip(
            field / scale,
            -1.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Gradient edge field
    # ---------------------------------------------------------

    def calculate_gradient_field(
        self,
        magnitude: np.ndarray,
    ) -> np.ndarray:
        normalized = self.normalize_field(
            magnitude
        )

        return np.clip(
            np.abs(normalized),
            0.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Laplacian
    # ---------------------------------------------------------

    def calculate_laplacian(
        self,
        luminance: np.ndarray,
    ) -> np.ndarray:
        laplacian = cv2.Laplacian(
            luminance,
            cv2.CV_32F,
            ksize=3,
        )

        return laplacian.astype(np.float32)

    # ---------------------------------------------------------
    # Multi-scale response
    # ---------------------------------------------------------

    def calculate_scale_response(
        self,
        luminance: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Extract edge structure at a specific spatial scale.

        Difference between original luminance and
        Gaussian-smoothed luminance.
        """

        kernel_size = max(
            3,
            int(round(sigma * 6 + 1)),
        )

        if kernel_size % 2 == 0:
            kernel_size += 1

        blurred = cv2.GaussianBlur(
            luminance,
            (kernel_size, kernel_size),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        response = np.abs(
            luminance - blurred
        )

        return self.normalize_field(
            response
        ).astype(np.float32)

    def calculate_multiscale_response(
        self,
        luminance: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        small = self.calculate_scale_response(
            luminance,
            self.style.edge_sigma_small,
        )

        medium = self.calculate_scale_response(
            luminance,
            self.style.edge_sigma_medium,
        )

        large = self.calculate_scale_response(
            luminance,
            self.style.edge_sigma_large,
        )

        # The multi-scale field itself is an equal
        # combination of the three spatial scales.
        multiscale = (
            small
            + medium
            + large
        ) / 3.0

        return (
            small,
            medium,
            large,
            np.clip(
                multiscale,
                0.0,
                1.0,
            ).astype(np.float32),
        )

    # ---------------------------------------------------------
    # Soft threshold
    # ---------------------------------------------------------

    @staticmethod
    def sigmoid(
        x: np.ndarray,
    ) -> np.ndarray:
        x = np.clip(
            x,
            -60.0,
            60.0,
        )

        return (
            1.0
            / (1.0 + np.exp(-x))
        ).astype(np.float32)

    def soft_threshold(
        self,
        field: np.ndarray,
    ) -> np.ndarray:
        threshold = float(
            self.style.edge_threshold
        )

        softness = max(
            float(self.style.edge_softness),
            1e-5,
        )

        response = (
            field - threshold
        ) / softness

        return self.sigmoid(
            response
        )

    # ---------------------------------------------------------
    # Combined edge probability
    # ---------------------------------------------------------

    def calculate_edge_probability(
        self,
        gradient_field: np.ndarray,
        laplacian: np.ndarray,
        multiscale: np.ndarray,
    ) -> np.ndarray:

        laplacian_field = np.abs(
            self.normalize_field(
                laplacian
            )
        )

        wg = float(
            self.style.edge_gradient_weight
        )

        wl = float(
            self.style.edge_laplacian_weight
        )

        wm = float(
            self.style.edge_multiscale_weight
        )

        total = wg + wl + wm

        if total <= 0:
            raise ValueError(
                "Edge field weights must sum to > 0"
            )

        combined = (
            wg * gradient_field
            + wl * laplacian_field
            + wm * multiscale
        ) / total

        combined = np.clip(
            combined,
            0.0,
            1.0,
        )

        return self.soft_threshold(
            combined
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Line strength
    # ---------------------------------------------------------

    def calculate_line_strength(
        self,
        edge_probability: np.ndarray,
    ) -> np.ndarray:

        strength = (
            edge_probability
            * float(self.style.edge_strength)
        )

        strength = np.clip(
            strength,
            float(self.style.line_min_strength),
            float(self.style.line_max_strength),
        )

        return strength.astype(np.float32)

    # ---------------------------------------------------------
    # Line field
    # ---------------------------------------------------------

    def calculate_line_field(
        self,
        line_strength: np.ndarray,
    ) -> np.ndarray:

        darkness = float(
            self.style.line_darkness
        )

        line_field = (
            1.0
            - line_strength * darkness
        )

        return np.clip(
            line_field,
            0.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Apply lines
    # ---------------------------------------------------------

    def apply_line_field(
        self,
        rgb: np.ndarray,
        line_field: np.ndarray,
    ) -> np.ndarray:

        output = (
            rgb
            * line_field[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Preserve highlights
    # ---------------------------------------------------------

    def preserve_highlights(
        self,
        original: np.ndarray,
        output: np.ndarray,
        luminance: np.ndarray,
    ) -> np.ndarray:

        threshold = 0.80

        highlight = np.clip(
            (luminance - threshold)
            / max(
                1.0 - threshold,
                1e-5,
            ),
            0.0,
            1.0,
        )

        preservation = (
            highlight
            * float(
                self.style.line_preserve_highlights
            )
        )

        output = (
            output * (1.0 - preservation[..., None])
            + original * preservation[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Preserve deep shadows
    # ---------------------------------------------------------

    def preserve_shadows(
        self,
        original: np.ndarray,
        output: np.ndarray,
        luminance: np.ndarray,
    ) -> np.ndarray:

        threshold = 0.20

        shadow = np.clip(
            (threshold - luminance)
            / max(
                threshold,
                1e-5,
            ),
            0.0,
            1.0,
        )

        preservation = (
            shadow
            * float(
                self.style.line_preserve_shadows
            )
        )

        output = (
            output * (1.0 - preservation[..., None])
            + original * preservation[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        ).astype(np.float32)

    # ---------------------------------------------------------
    # Full transformation
    # ---------------------------------------------------------

    def transform(
        self,
        frame_rgb: np.ndarray,
    ) -> EdgeFieldResult:

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
            gx,
            gy,
            magnitude,
            orientation,
        ) = self.calculate_gradient(
            luminance
        )

        gradient_field = (
            self.calculate_gradient_field(
                magnitude
            )
        )

        laplacian = (
            self.calculate_laplacian(
                luminance
            )
        )

        (
            small,
            medium,
            large,
            multiscale,
        ) = self.calculate_multiscale_response(
            luminance
        )

        edge_probability = (
            self.calculate_edge_probability(
                gradient_field,
                laplacian,
                multiscale,
            )
        )

        line_strength = (
            self.calculate_line_strength(
                edge_probability
            )
        )

        line_field = (
            self.calculate_line_field(
                line_strength
            )
        )

        output = self.apply_line_field(
            rgb,
            line_field,
        )

        output = self.preserve_highlights(
            rgb,
            output,
            luminance,
        )

        output = self.preserve_shadows(
            rgb,
            output,
            luminance,
        )

        output_rgb = np.clip(
            output * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8)

        return EdgeFieldResult(
            output_rgb=output_rgb,
            input_rgb=(
                np.clip(
                    rgb * 255.0,
                    0.0,
                    255.0,
                ).round().astype(np.uint8)
            ),
            luminance=luminance,
            gradient_x=gx,
            gradient_y=gy,
            gradient_magnitude=magnitude,
            gradient_orientation=orientation,
            laplacian=laplacian,
            small_scale_response=small,
            medium_scale_response=medium,
            large_scale_response=large,
            multiscale_response=multiscale,
            edge_probability=edge_probability,
            line_strength=line_strength,
            line_field=line_field,
        )

    def render(
        self,
        frame_rgb: np.ndarray,
    ) -> np.ndarray:
        return self.transform(
            frame_rgb
        ).output_rgb


def compute_edge_field(
    lit_field: np.ndarray,
    luminance: np.ndarray,
    style: MathematicalAnimeStyle,
    edge_mask_modifier: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compatibility function for pipeline compositor:
        Gx = dY/dx, Gy = dY/dy, G = sqrt(Gx^2 + Gy^2)
        L = |nabla^2 Y|
        E = 0.70 * G + 0.30 * L
        E_A = sigma((E - T_e) / K_e)
        I_L = alpha * line_darkness * E_A
        C_line = (1 - I_L) * C_H + I_L * C_ink
    Returns:
        (stylized_with_ink, edge_intensity, raw_edge_field)
    """
    Gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(Gx ** 2 + Gy ** 2)

    laplacian = np.abs(cv2.Laplacian(luminance, cv2.CV_32F, ksize=3))

    max_g = max(1e-5, float(np.percentile(grad_mag, 99.5)))
    max_l = max(1e-5, float(np.percentile(laplacian, 99.5)))
    g_norm = np.clip(grad_mag / max_g, 0.0, 1.0)
    l_norm = np.clip(laplacian / max_l, 0.0, 1.0)

    E = 0.70 * g_norm + 0.30 * l_norm

    if edge_mask_modifier is not None:
        E = E * edge_mask_modifier

    Te = style.edge_threshold
    Ke = max(1e-4, style.edge_softness)
    z = np.clip((E - Te) / Ke, -20.0, 20.0)
    edge_prob = 1.0 / (1.0 + np.exp(-z))
    edge_prob = edge_prob[:, :, np.newaxis]

    ink_intensity = np.clip(style.edge_strength * style.line_darkness * edge_prob, 0.0, 1.0)

    ink_rgb = np.array(style.ink_color, dtype=np.float32) / 255.0

    stylized_ink = (1.0 - ink_intensity) * lit_field + ink_intensity * ink_rgb
    stylized_ink = np.clip(stylized_ink, 0.0, 1.0)

    return stylized_ink, ink_intensity, E
