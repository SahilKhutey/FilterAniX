"""Stage 2: Anime Luminance Model and Tonal S-Curve Transformation."""
from __future__ import annotations

from typing import Tuple
import numpy as np

from .config import MathematicalAnimeStyle


def compute_tone_field(
    color_field: np.ndarray,
    style: MathematicalAnimeStyle,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes anime luminance model:
        Y = 0.299 R + 0.587 G + 0.114 B
        Y_1 = clip((Y - 0.5) * contrast + 0.5, 0.0, 1.0)
        Y_2 = Y_1 ^ gamma
        Y_A = 0.5 + (Y_2 - 0.5) * (1 + k * T)
    Returns:
        (Y_original, Y_anime, toned_color_field): All float32 in [0.0, 1.0].
    """
    # Rec.601 luminance
    Y = 0.299 * color_field[:, :, 0] + 0.587 * color_field[:, :, 1] + 0.114 * color_field[:, :, 2]
    Y = np.clip(Y, 0.0, 1.0)

    # Contrast S-curve
    Y_1 = np.clip((Y - 0.5) * style.contrast + 0.5, 0.0, 1.0)

    # Gamma compression / expansion
    Y_2 = np.power(np.maximum(Y_1, 1e-6), style.gamma)

    # Anime tone curve: clean midtones with subtle artistic expansion
    k = style.tone_artistic_factor
    T = style.tone_strength
    Y_A = np.clip(0.5 + (Y_2 - 0.5) * (1.0 + k * T), 0.0, 1.0)

    # Modulate color channels by luminance ratio
    tone_ratio = (Y_A / np.maximum(Y, 1e-5))[:, :, np.newaxis]
    toned_field = np.clip(
        color_field * (1.0 + style.tone_strength * (tone_ratio - 1.0)),
        0.0,
        1.0,
    )

    return Y, Y_A, toned_field
