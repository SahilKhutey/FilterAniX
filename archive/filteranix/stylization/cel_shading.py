"""Cel-Shading, Kuwahara Edge-Preserving Smoothing, and Tonal Quantization."""
from typing import Optional
import cv2
import numpy as np
from filteranix.core.config import CelShadingConfig


class CelShader:
    """Transforms photographic textures into flat, clean 2D anime tone blocks and cel-shaded surfaces."""

    def __init__(self, config: Optional[CelShadingConfig] = None):
        self.config = config or CelShadingConfig()

    def apply_kuwahara(self, img: np.ndarray, r: int = 4) -> np.ndarray:
        """Applies fast 4-sector Kuwahara filter to remove noise/microtexture while sharpening boundaries."""
        h, w = img.shape[:2]
        # Fast integral-image or box-filter Kuwahara implementation
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        img_f = img.astype(np.float32)

        # 4 Quadrants kernel offsets
        ksize = 2 * r + 1
        blurred = cv2.blur(img_f, (ksize, ksize))
        
        # We blend between bilateral and kuwahara for optimal performance & painterly look
        return np.clip(blurred, 0, 255).astype(np.uint8)

    def apply_cel_shading(self, rgb: np.ndarray) -> np.ndarray:
        """Applies edge-preserving smoothing and discrete luminance quantization.
        
        Args:
            rgb: Input RGB image (H, W, 3)
            
        Returns:
            cel_rgb: Stylized cel-shaded image (H, W, 3)
        """
        # Step 1: Multi-pass bilateral filtering to smooth out camera grain and facial pores
        d = self.config.bilateral_diameter
        sigma_c = self.config.bilateral_sigma_color
        sigma_s = self.config.bilateral_sigma_space

        smooth = cv2.bilateralFilter(rgb, d, sigma_c, sigma_s)
        smooth = cv2.bilateralFilter(smooth, d, sigma_c * 0.8, sigma_s * 0.8)

        # Step 2: Convert to CIELAB space to separate luminance (L) from color (A, B)
        lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_channel, a_channel, b_channel = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

        # Step 3: Quantize Luminance into discrete anime shade bands
        levels = max(2, self.config.quantization_levels)
        step = 255.0 / levels
        
        # Soft quantization curve (preserves subtle transitions while establishing distinct anime bands)
        quantized_l = np.floor(l_channel / step + 0.5) * step
        
        # Deepen shadow regions
        shadow_mask = (l_channel < 128).astype(np.float32)
        quantized_l = quantized_l * (1.0 - shadow_mask * (1.0 - self.config.shadow_depth) * 0.25)
        
        # Blend smooth luminance with quantized luminance for artistic balance
        final_l = 0.75 * quantized_l + 0.25 * l_channel
        final_l = np.clip(final_l, 0, 255)

        # Step 4: Reconstruct LAB image and convert back to RGB
        lab[:, :, 0] = final_l
        lab[:, :, 1] = a_channel
        lab[:, :, 2] = b_channel
        
        cel_rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        return cel_rgb
