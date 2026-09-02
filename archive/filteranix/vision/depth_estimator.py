"""Relative Depth Estimation for Depth-Aware Line Weights and Volumetric Shading."""
from typing import Optional
import cv2
import numpy as np


class DepthEstimator:
    """Estimates relative depth maps for atmospheric haze, depth-of-field, and line-weight modulation."""

    def __init__(self):
        pass

    def estimate_depth(self, rgb: np.ndarray, person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Computes relative depth map normalized to [0.0, 1.0] (1.0 = closest to camera, 0.0 = background)."""
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Multi-scale gradient and atmospheric luminance prior
        # Objects closer to camera typically exhibit higher contrast and sharper gradients
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        laplacian_smooth = cv2.GaussianBlur(laplacian, (31, 31), 0)
        grad_norm = laplacian_smooth / (np.max(laplacian_smooth) + 1e-5)

        # Distance from bottom-center prior (typical creator perspective camera setup)
        y, x = np.ogrid[:h, :w]
        perspective_prior = (y / float(h)).astype(np.float32)

        depth = 0.4 * grad_norm + 0.3 * perspective_prior
        if person_mask is not None:
            depth = depth * 0.3 + person_mask * 0.7

        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth) + 1e-5)
        return np.clip(depth, 0.0, 1.0).astype(np.float32)
