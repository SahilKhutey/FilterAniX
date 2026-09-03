"""Stage 5: Warm Highlight Field and Specular Cel Planes."""
from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import MathematicalAnimeStyle


def compute_highlight_field(
    shaded_field: np.ndarray,
    luminance: np.ndarray,
    style: MathematicalAnimeStyle,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes continuous highlight field:
        H(x,y) = sigma((Y(x,y) - T_h) / k_h)
        C_H = C_S + alpha_h * H * K_warm
    Returns:
        (lit_color, highlight_field): lit_color in [0.0, 1.0] RGB, highlight_field (H, W, 1).
    """
    Th = style.highlight_threshold
    kh = max(1e-4, style.highlight_softness)

    # Sigmoid highlight probability
    z = np.clip((luminance - Th) / kh, -20.0, 20.0)
    highlight_prob = 1.0 / (1.0 + np.exp(-z))
    highlight_field = highlight_prob[:, :, np.newaxis]

    alpha_h = style.highlight_strength
    warm_tint = np.array(style.warm_light_color, dtype=np.float32)

    lit = np.clip(shaded_field + alpha_h * highlight_field * warm_tint, 0.0, 1.0)

    return lit, highlight_field
