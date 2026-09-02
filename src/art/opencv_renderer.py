from __future__ import annotations

from typing import Any, Optional
import cv2
import numpy as np

from src.art.types import StylePreset, ControlMap


class OpenCVArtRenderer:
    """Fast deterministic artistic renderer for preview, fallback, and intermediate frames."""

    def __init__(
        self,
        bilateral_passes: int = 2,
        edge_strength: float = 0.65,
    ):
        self.bilateral_passes = bilateral_passes
        self.edge_strength = edge_strength

    def render(self, frame: np.ndarray, *args, **kwargs) -> np.ndarray:
        result = frame.copy()

        for _ in range(self.bilateral_passes):
            result = cv2.bilateralFilter(
                result,
                d=7,
                sigmaColor=50,
                sigmaSpace=50,
            )

        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result

        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2,
        )

        if len(result.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        output = cv2.bitwise_and(
            result,
            edges,
        )

        return cv2.addWeighted(
            result,
            1.0 - self.edge_strength,
            output,
            self.edge_strength,
            0,
        )


class OpenCVIllustrationRenderer:
    """High-precision procedural anime illustration renderer with Reinhard reference palette alignment."""

    def __init__(self, style_preset: Optional[StylePreset] = None):
        self.style = style_preset or StylePreset()

    def apply_color_transfer(self, source_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
        """Transfers the color distribution of the reference image to the source image in Lab space."""
        src_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

        src_mean, src_std = np.mean(src_lab, axis=(0, 1)), np.std(src_lab, axis=(0, 1)) + 1e-5
        ref_mean, ref_std = np.mean(ref_lab, axis=(0, 1)), np.std(ref_lab, axis=(0, 1)) + 1e-5

        trans_lab = ((src_lab - src_mean) / src_std) * (0.5 * ref_std + 0.5 * src_std) + (0.5 * ref_mean + 0.5 * src_mean)
        trans_lab = np.clip(trans_lab, 0, 255).astype(np.uint8)

        return cv2.cvtColor(trans_lab, cv2.COLOR_LAB2RGB)

    def extract_line_art(self, rgb: np.ndarray, edge_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts stylized dark ink lines on white background."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        g1 = cv2.GaussianBlur(gray, (0, 0), 0.8)
        g2 = cv2.GaussianBlur(gray, (0, 0), 1.6)
        dog = g1 - 0.98 * g2
        diff = dog - (-0.1)
        edge_resp = np.where(diff >= 0, 1.0, 1.0 + np.tanh(10.0 * diff))
        edge_resp = np.clip(edge_resp, 0.0, 1.0)

        line_intensity = (1.0 - edge_resp) * self.style.line_weight
        line_intensity = np.clip(line_intensity, 0.0, 1.0)

        tint = np.array(self.style.line_tint, dtype=np.float32) / 255.0
        ink_rgb = (1.0 - line_intensity[..., np.newaxis]) + line_intensity[..., np.newaxis] * tint
        return np.clip(ink_rgb * 255.0, 0, 255).astype(np.uint8)

    def render(
        self,
        rgb: np.ndarray,
        control_map: Optional[ControlMap] = None,
        vision_data: Optional[Any] = None,
        reference_rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Transforms a raw frame into an anime illustration."""
        h, w = rgb.shape[:2]

        canvas = rgb
        if reference_rgb is not None:
            canvas = self.apply_color_transfer(canvas, reference_rgb)

        smooth = cv2.bilateralFilter(canvas, 9, 65, 45)
        smooth = cv2.bilateralFilter(smooth, 9, 50, 35)

        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        step = 255.0 / max(2, self.style.shading_levels)
        quantized_l = np.floor(l_chan / step + 0.5) * step

        shadow_mask = (l_chan < 128).astype(np.float32)
        quantized_l = quantized_l * (1.0 - shadow_mask * 0.15)

        lab[:, :, 0] = np.clip(0.75 * quantized_l + 0.25 * l_chan, 0, 255)
        cel = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

        gray = cv2.cvtColor(cel.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        key_tint = np.array(self.style.key_light_color, dtype=np.float32)
        shadow_tint = np.array(self.style.shadow_color, dtype=np.float32)

        high_w = (gray ** 1.5)[..., np.newaxis] * self.style.color_warmth * 0.25
        shad_w = ((1.0 - gray) ** 1.5)[..., np.newaxis] * self.style.shadow_coolness * 0.20

        cel = cel * (1.0 - high_w) + key_tint * high_w
        cel = cel * (1.0 - shad_w) + shadow_tint * shad_w

        edge_m = control_map.edge_map if control_map else None
        line_art = self.extract_line_art(rgb, edge_map=edge_m)
        line_f = line_art.astype(np.float32) / 255.0
        composite = cel * line_f

        if self.style.saturation_boost != 1.0:
            hsv = cv2.cvtColor(np.clip(composite, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.style.saturation_boost, 0, 255)
            composite = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        kernel_x = cv2.getGaussianKernel(w, w * 0.75)
        kernel_y = cv2.getGaussianKernel(h, h * 0.75)
        vignette = (kernel_y * kernel_x.T)
        vignette = vignette / np.max(vignette)
        composite = composite * (0.80 + 0.20 * vignette[..., np.newaxis])

        return np.clip(composite, 0, 255).astype(np.uint8)
