"""Stage 6: Anime Line-Art and Charcoal Ink Field."""
from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np

from .config import MathematicalAnimeStyle


def compute_edge_field(
    lit_field: np.ndarray,
    luminance: np.ndarray,
    style: MathematicalAnimeStyle,
    edge_mask_modifier: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes smooth anime line-art field using gradient and Laplacian operators:
        Gx = dY/dx, Gy = dY/dy, G = sqrt(Gx^2 + Gy^2)
        L = |nabla^2 Y|
        E = 0.70 * G + 0.30 * L
        E_A = sigma((E - T_e) / K_e)
        I_L = alpha * line_darkness * E_A
        C_line = (1 - I_L) * C_H + I_L * C_ink
    Returns:
        (stylized_with_ink, edge_intensity, raw_edge_field):
        stylized_with_ink in [0.0, 1.0] RGB, edge_intensity (H, W, 1), raw_edge_field (H, W).
    """
    # Sobel first-order spatial derivatives
    Gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(Gx ** 2 + Gy ** 2)

    # Laplacian second-order isotropic derivative
    laplacian = np.abs(cv2.Laplacian(luminance, cv2.CV_32F, ksize=3))

    # Normalize response
    max_g = max(1e-5, float(np.percentile(grad_mag, 99.5)))
    max_l = max(1e-5, float(np.percentile(laplacian, 99.5)))
    g_norm = np.clip(grad_mag / max_g, 0.0, 1.0)
    l_norm = np.clip(laplacian / max_l, 0.0, 1.0)

    # Combined edge response
    E = 0.70 * g_norm + 0.30 * l_norm

    # Optional spatial modulation (e.g., hair/eye edge boost or face edge control)
    if edge_mask_modifier is not None:
        E = E * edge_mask_modifier

    # Smooth sigmoidal edge activation
    Te = style.edge_threshold
    Ke = max(1e-4, style.edge_softness)
    z = np.clip((E - Te) / Ke, -20.0, 20.0)
    edge_prob = 1.0 / (1.0 + np.exp(-z))
    edge_prob = edge_prob[:, :, np.newaxis]

    # Effective ink intensity
    ink_intensity = np.clip(style.edge_strength * style.line_darkness * edge_prob, 0.0, 1.0)

    # Warm charcoal / dark brown anime ink vector (not synthetic pure black)
    ink_rgb = np.array(style.ink_color, dtype=np.float32) / 255.0

    # Alpha composition
    stylized_ink = (1.0 - ink_intensity) * lit_field + ink_intensity * ink_rgb
    stylized_ink = np.clip(stylized_ink, 0.0, 1.0)

    return stylized_ink, ink_intensity, E
