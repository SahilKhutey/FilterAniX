"""Stage: Geometry Field and Surface Normal Estimation."""
from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


def compute_surface_normals(
    luminance: np.ndarray,
    depth_scale: float = 2.0,
) -> np.ndarray:
    """
    Approximates 3D surface normal vector field from luminance gradients:
        Gx = dY/dx, Gy = dY/dy
        N(x,y) = (-Gx * scale, -Gy * scale, 1.0) / sqrt((Gx*scale)^2 + (Gy*scale)^2 + 1.0)
    Returns:
        normals: (H, W, 3) float32 unit vectors.
    """
    Gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)

    nx = -Gx * depth_scale
    ny = -Gy * depth_scale
    nz = np.ones_like(luminance, dtype=np.float32)

    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    norm = np.maximum(norm, 1e-6)

    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return normals
