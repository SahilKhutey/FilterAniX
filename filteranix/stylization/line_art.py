"""Dynamic Line Art Generation via Extended Difference of Gaussians (XDoG)."""
from typing import List, Optional, Tuple
import cv2
import numpy as np
from filteranix.core.config import LineArtConfig


class LineArtExtractor:
    """Extracts clean, expressive, stylized anime line art using XDoG and adaptive edge filtering."""

    def __init__(self, config: Optional[LineArtConfig] = None):
        self.config = config or LineArtConfig()

    def extract_xdog(self, rgb: np.ndarray, depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies Extended Difference of Gaussians (XDoG) filter to generate smooth, ink-like contours.
        
        Args:
            rgb: Input RGB image (H, W, 3)
            depth_map: Optional depth map (H, W) for depth-adaptive line thickness
            
        Returns:
            line_art_rgb: RGB image (H, W, 3) where lines are colored with config.line_color_tint on white/transparent
        """
        # Convert to float grayscale [0.0, 1.0]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        sigma = max(0.2, self.config.sigma)
        k_sigma = max(1.1, self.config.k_sigma)
        gamma = self.config.gamma
        epsilon = self.config.epsilon
        phi = self.config.phi

        # Compute Gaussian blurs at two scales
        g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k_sigma)

        # Difference of Gaussians
        dog = g1 - gamma * g2

        # Soft thresholding (tanh activation)
        diff = dog - epsilon
        edge_response = np.where(diff >= 0, 1.0, 1.0 + np.tanh(phi * diff))
        edge_response = np.clip(edge_response, 0.0, 1.0)

        # Depth-dependent line weight modulation if depth map is provided
        if depth_map is not None:
            # Foregrounds get bolder lines, backgrounds get finer lines
            weight_factor = 0.8 + 0.6 * depth_map
            edge_response = np.power(edge_response, weight_factor * self.config.line_weight)
        elif self.config.line_weight != 1.0:
            edge_response = np.power(edge_response, self.config.line_weight)

        # Invert so lines = 1.0 (ink), paper = 0.0
        line_intensity = 1.0 - edge_response
        line_intensity = np.clip(line_intensity, 0.0, 1.0)

        # Map to styled ink color tint
        tint = np.array(self.config.line_color_tint, dtype=np.float32) / 255.0
        # Background is white (1,1,1), lines blend towards tint
        ink_rgb = (1.0 - line_intensity[..., np.newaxis]) + line_intensity[..., np.newaxis] * tint
        ink_rgb = np.clip(ink_rgb * 255.0, 0, 255).astype(np.uint8)

        return ink_rgb

    def extract_contour_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Returns a binary or continuous [0.0, 1.0] line mask for alpha blending."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g1 = cv2.GaussianBlur(gray, (0, 0), self.config.sigma)
        g2 = cv2.GaussianBlur(gray, (0, 0), self.config.sigma * self.config.k_sigma)
        dog = g1 - self.config.gamma * g2
        diff = dog - self.config.epsilon
        edge = np.where(diff >= 0, 1.0, 1.0 + np.tanh(self.config.phi * diff))
        return (1.0 - np.clip(edge, 0.0, 1.0)).astype(np.float32)
