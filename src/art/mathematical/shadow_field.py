"""Stage 4: Continuous Cel-Shadow Field and Tonal Depths."""
from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import MathematicalAnimeStyle


def compute_shadow_field(
    color_field: np.ndarray,
    luminance: np.ndarray,
    style: MathematicalAnimeStyle,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes continuous cel-shadow field:
        S(x,y) = sigma((T_s - Y(x,y)) / k_s)
        C_S = C_A * (1 - alpha_s * S) + cool_tint_shift
    Returns:
        (shaded_color, shadow_field): shaded_color in [0.0, 1.0] RGB, shadow_field (H, W, 1).
    """
    Ts = style.shadow_threshold
    ks = max(1e-4, style.shadow_softness)

    # Sigmoid shadow probability field
    z = np.clip((Ts - luminance) / ks, -20.0, 20.0)
    shadow_prob = 1.0 / (1.0 + np.exp(-z))
    shadow_field = shadow_prob[:, :, np.newaxis]

    alpha_s = style.shadow_strength
    # Subtle cinematic cool tint shift for deep cel shadows
    cool_shadow_tint = np.array([-0.02, -0.01, 0.03], dtype=np.float32)

    shaded = color_field * (1.0 - alpha_s * shadow_field)
    shaded = np.clip(shaded + shadow_field * cool_shadow_tint * alpha_s, 0.0, 1.0)

    return shaded, shadow_field
