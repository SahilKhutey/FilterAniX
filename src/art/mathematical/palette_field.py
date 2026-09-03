"""Stage 3: Soft Color Quantization and Continuous Palette Projection."""
from __future__ import annotations

from typing import Optional
import numpy as np

from .config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle


def compute_palette_projection(
    toned_field: np.ndarray,
    style: MathematicalAnimeStyle,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Continuous soft projection onto the 7-tier anime color palette:
        d_k(x,y) = ||C(x,y) - P_k||^2
        w_k(x,y) = exp(-d_k / tau) / sum_j exp(-d_j / tau)
        C_P(x,y) = sum_k w_k * P_k
        C_A = (1 - alpha) * C + alpha * C_P
    Returns:
        C_A in [0.0, 1.0] float32 RGB.
    """
    if style.palette_mix <= 0.0:
        return toned_field.copy()

    pal = palette if palette is not None else DEFAULT_ANIME_PALETTE
    # Normalize palette to [0.0, 1.0] float32 if needed
    if pal.max() > 1.0:
        pal = pal / 255.0
    pal = pal.astype(np.float32)

    h, w, c = toned_field.shape
    flat_pixels = toned_field.reshape(-1, 3)  # (N, 3)

    # Vectorized Euclidean distance computation
    # ||p - c_k||^2 = ||p||^2 + ||c_k||^2 - 2 (p . c_k)
    p_sq = np.sum(flat_pixels ** 2, axis=1, keepdims=True)  # (N, 1)
    c_sq = np.sum(pal ** 2, axis=1, keepdims=True).T       # (1, K)
    dot = np.dot(flat_pixels, pal.T)                        # (N, K)
    dists = np.maximum(0.0, p_sq + c_sq - 2.0 * dot)       # (N, K)

    # Softmax temperature
    tau = max(1e-3, 0.05 * style.palette_temperature)
    min_dists = np.min(dists, axis=1, keepdims=True)
    exp_weights = np.exp(-(dists - min_dists) / tau)
    weights = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)

    # Reconstruct continuous palette field
    palette_projected = np.dot(weights, pal).reshape(h, w, 3)

    # Smooth artistic blending
    alpha = style.palette_mix
    projected_field = np.clip(
        (1.0 - alpha) * toned_field + alpha * palette_projected,
        0.0,
        1.0,
    )

    return projected_field
