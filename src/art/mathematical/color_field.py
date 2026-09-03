"""Stage 1: Perceptual Color Field and Photographic Texture Suppression."""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np

from .config import MathematicalAnimeStyle


def compute_color_field(
    rgb_uint8: np.ndarray,
    style: MathematicalAnimeStyle,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms raw RGB frame into perceptual color field with photographic noise suppression:
        I(x,y) -> C(x,y)
        C_s = G_sigma(C)
        C' = C_s + alpha * (C - C_s)
    Returns:
        (color_field_float, smoothed_base_float): Both in [0.0, 1.0] float32 RGB.
    """
    img_f = rgb_uint8.astype(np.float32) / 255.0

    # Bilateral edge-preserving filtering on uint8 for high CPU throughput
    # Sigma values tuned for photographic noise removal without blurring crisp boundaries
    smooth_uint8 = cv2.bilateralFilter(
        rgb_uint8,
        d=7,
        sigmaColor=45,
        sigmaSpace=45,
    )
    smooth_f = smooth_uint8.astype(np.float32) / 255.0

    # Controlled detail retention (alpha = style.detail_retention)
    # Suppresses micro-camera sensor noise while keeping silhouette / clothing contrast
    detail = img_f - smooth_f
    color_field = np.clip(
        smooth_f + style.detail_retention * detail,
        0.0,
        1.0,
    )

    # Saturation enhancement in HSV color space
    if abs(style.saturation - 1.0) > 1e-4:
        hsv = cv2.cvtColor((color_field * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * style.saturation, 0.0, 255.0)
        color_field = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    return color_field, smooth_f
