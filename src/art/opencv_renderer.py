from __future__ import annotations

from typing import Any, Optional
import cv2
import numpy as np

from src.art.types import StylePreset, ControlMap


def render_anime_mouth(
    canvas_rgb: np.ndarray,
    face_data: Any,
    viseme: str = "closed",
    mouth_opening: float = 0.0,
    line_tint: tuple[int, int, int] = (40, 20, 30),
) -> np.ndarray:
    """Renders stylized anime mouth shape based on face landmarks / bounding box and viseme telemetry."""
    if not face_data or not hasattr(face_data, "landmarks") or not face_data.landmarks:
        return canvas_rgb

    h, w = canvas_rgb.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_data.landmarks]
    
    # Identify mouth center & dimensions
    if len(pts) >= 468:
        # MediaPipe lips: 13 (upper inner), 14 (lower inner), 61 (left corner), 291 (right corner)
        mouth_cx = (pts[61][0] + pts[291][0]) // 2
        mouth_cy = (pts[13][1] + pts[14][1]) // 2
        mouth_w = max(6, int(abs(pts[291][0] - pts[61][0]) * 0.9))
    elif hasattr(face_data, "bbox") and face_data.bbox:
        bx = int(face_data.bbox.x * w)
        by = int(face_data.bbox.y * h)
        bw = int(face_data.bbox.width * w)
        bh = int(face_data.bbox.height * h)
        mouth_cx = bx + bw // 2
        mouth_cy = by + int(bh * 0.78)
        mouth_w = max(6, int(bw * 0.28))
    else:
        return canvas_rgb

    ratio = max(0.0, min(1.0, mouth_opening))
    if viseme == "closed" or (viseme is None and ratio < 0.10):
        # Anime line mouth with slight natural curve
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, max(1, mouth_w // 10)), 0, 10, 170, line_tint, 2, cv2.LINE_AA)
    elif viseme == "slightly_open" or ratio < 0.22:
        mouth_h = max(2, int(mouth_w * 0.22))
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, (50, 20, 30), -1, cv2.LINE_AA)
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, line_tint, 2, cv2.LINE_AA)
    elif viseme == "open" or ratio < 0.40:
        mouth_h = max(4, int(mouth_w * 0.45))
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, (80, 25, 45), -1, cv2.LINE_AA)
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy + mouth_h // 3), (mouth_w // 3, mouth_h // 3), 0, 0, 180, (180, 80, 100), -1, cv2.LINE_AA)
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, line_tint, 2, cv2.LINE_AA)
    else:  # wide_open
        mouth_h = max(6, int(mouth_w * 0.70))
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, (90, 20, 40), -1, cv2.LINE_AA)
        cv2.line(canvas_rgb, (mouth_cx - mouth_w // 3, mouth_cy - mouth_h // 3), (mouth_cx + mouth_w // 3, mouth_cy - mouth_h // 3), (240, 240, 245), 2, cv2.LINE_AA)
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy + mouth_h // 3), (mouth_w // 3, mouth_h // 3), 0, 0, 180, (200, 90, 110), -1, cv2.LINE_AA)
        cv2.ellipse(canvas_rgb, (mouth_cx, mouth_cy), (mouth_w // 2, mouth_h), 0, 0, 360, line_tint, 2, cv2.LINE_AA)

    return canvas_rgb


class FastPreviewRenderer:
    """
    ⚡ FAST PREVIEW RENDERER
    Approximate visualization - Not final production rendering.
    Lightweight fallback providing fast 30 FPS preview with zero GPU overhead.
    For full deterministic continuous image fields with character semantic preservation,
    use MathematicalAnimeEngine.
    """

    def __init__(
        self,
        bilateral_passes: int = 2,
        edge_strength: float = 0.65,
    ):
        self.bilateral_passes = bilateral_passes
        self.edge_strength = edge_strength

    def render(
        self,
        frame: np.ndarray,
        vision_data: Optional[Any] = None,
        lipsync_record: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> np.ndarray:
        result = frame.copy()

        # Bilateral smoothing to preserve cartoon color regions
        for _ in range(self.bilateral_passes):
            result = cv2.bilateralFilter(
                result,
                d=7,
                sigmaColor=50,
                sigmaSpace=50,
            )

        # 1. Correct RGB -> Grayscale conversion (Fix color-space bug)
        if len(result.shape) == 3 and result.shape[2] == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        else:
            gray = result.copy()

        # 2. Warm anime luminance enhancement (avoid crushed blacks)
        gray_f = gray.astype(np.float32) / 255.0
        # Gentle anime S-curve tone mapping
        toned_gray = np.clip(1.0 / (1.0 + np.exp(-4.5 * (gray_f - 0.45))), 0.0, 1.0)
        toned_uint8 = (toned_gray * 255.0).astype(np.uint8)

        # 3. Clean anime linework without speckled adaptive-threshold noise
        # Compute smooth gradient magnitude
        grad_x = cv2.Sobel(toned_uint8, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(toned_uint8, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        grad_norm = np.clip(grad_mag / 255.0 * 2.5, 0.0, 1.0)
        grad_norm = cv2.GaussianBlur(grad_norm, (3, 3), 0.5)

        # Soft anime ink lines (warm dark purple-brown ink: [32, 27, 34])
        ink_color = np.array([32.0, 27.0, 34.0], dtype=np.float32)
        line_weight = np.clip(grad_norm * self.edge_strength, 0.0, 1.0)

        # Blend smoothed color base with warm anime ink
        result_f = result.astype(np.float32)
        # Apply slight warm anime color grading
        result_f[:, :, 0] = np.clip(result_f[:, :, 0] * 1.04 + 4.0, 0.0, 255.0)  # warm red/skin
        result_f[:, :, 1] = np.clip(result_f[:, :, 1] * 1.01 + 2.0, 0.0, 255.0)  # green
        result_f[:, :, 2] = np.clip(result_f[:, :, 2] * 0.98 - 1.0, 0.0, 255.0)  # reduce harsh blue

        ink_broadcast = np.tile(ink_color.reshape((1, 1, 3)), (result.shape[0], result.shape[1], 1))
        weight_broadcast = line_weight[:, :, np.newaxis]

        blended_f = result_f * (1.0 - weight_broadcast) + ink_broadcast * weight_broadcast
        blended = np.clip(blended_f, 0.0, 255.0).astype(np.uint8)

        # Apply character anime mouth rendering if vision face or lipsync present
        if vision_data and hasattr(vision_data, "faces") and vision_data.faces:
            face = vision_data.faces[0]
            viseme = getattr(lipsync_record, "viseme", "closed") if lipsync_record else "closed"
            ratio = getattr(lipsync_record, "mouth_open_ratio", getattr(face, "mouth_opening", 0.0))
            blended = render_anime_mouth(blended, face, viseme=viseme, mouth_opening=ratio)

        return blended


class OpenCVIllustrationRenderer:
    """High-precision procedural anime illustration renderer with Reinhard reference palette alignment."""

    def __init__(self, style_preset: Optional[StylePreset | StyleConfig] = None):
        if style_preset is None:
            self.style = StylePreset()
        elif hasattr(style_preset, "style") and isinstance(style_preset.style, StylePreset):
            self.style = style_preset.style
        elif isinstance(style_preset, StylePreset):
            self.style = style_preset
        else:
            self.style = StylePreset()

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
        lipsync_record: Optional[Any] = None,
    ) -> np.ndarray:
        """Transforms a raw frame into an anime illustration with lip-sync viseme animation."""
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

        out_frame = np.clip(composite, 0, 255).astype(np.uint8)

        # Apply anime mouth rendering based on face and lipsync viseme
        if vision_data and hasattr(vision_data, "faces") and vision_data.faces:
            face = vision_data.faces[0]
            viseme = getattr(lipsync_record, "viseme", "closed") if lipsync_record else "closed"
            ratio = getattr(lipsync_record, "mouth_open_ratio", getattr(face, "mouth_opening", 0.0))
            out_frame = render_anime_mouth(out_frame, face, viseme=viseme, mouth_opening=ratio, line_tint=self.style.line_tint)

        return out_frame


# Backwards compatibility alias
OpenCVArtRenderer = FastPreviewRenderer
