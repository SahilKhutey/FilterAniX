"""Multi-Layer Compositor for Anime/Illustration Video."""
from typing import Optional
import cv2
import numpy as np
from filteranix.core.config import CompositorConfig


class FrameCompositor:
    """Blends foreground, background, dynamic line art, and cinematic lighting into a polished illustrated frame."""

    def __init__(self, config: Optional[CompositorConfig] = None):
        self.config = config or CompositorConfig()

    def composite(
        self,
        foreground_rgb: np.ndarray,
        background_rgb: np.ndarray,
        person_mask: np.ndarray,
        line_art_rgb: np.ndarray,
    ) -> np.ndarray:
        """Combines all visual layers into the final stylized output frame.
        
        Args:
            foreground_rgb: Stylized creator layer (H, W, 3)
            background_rgb: Stylized static background layer (H, W, 3)
            person_mask: Alpha mask for person [0.0, 1.0] (H, W)
            line_art_rgb: Ink line art image (H, W, 3) where paper is 255 and lines are dark tint
            
        Returns:
            final_frame: Broadcast-ready composite frame (H, W, 3), uint8
        """
        h, w = foreground_rgb.shape[:2]
        
        # Step 1: Alpha Blend Foreground Person onto Background Canvas
        # Soften mask boundary with a small edge-preserving blur
        alpha = cv2.GaussianBlur(person_mask, (5, 5), 0)[..., np.newaxis]
        alpha = np.clip(alpha, 0.0, 1.0)
        
        base_composite = (alpha * foreground_rgb.astype(np.float32) + (1.0 - alpha) * background_rgb.astype(np.float32))

        # Step 2: Ink Line Art Blend (Multiply Mode)
        if self.config.line_opacity > 0:
            line_f = line_art_rgb.astype(np.float32) / 255.0
            # Multiply: base * line
            opacity = self.config.line_opacity
            line_blended = base_composite * (1.0 - opacity + opacity * line_f)
            base_composite = np.clip(line_blended, 0, 255)

        # Step 3: Optional Subtle Cinematic Vignette
        if self.config.add_cinematic_vignette:
            base_composite = self._apply_vignette(base_composite, strength=self.config.vignette_strength)

        return np.clip(base_composite, 0, 255).astype(np.uint8)

    def _apply_vignette(self, img_f: np.ndarray, strength: float = 0.25) -> np.ndarray:
        """Gently darkens edges to focus attention on creator and desk setup."""
        h, w = img_f.shape[:2]
        kernel_x = cv2.getGaussianKernel(w, w * 0.7)
        kernel_y = cv2.getGaussianKernel(h, h * 0.7)
        kernel = kernel_y * kernel_x.T
        mask = kernel / np.max(kernel)
        vignette_map = 1.0 - strength * (1.0 - mask[..., np.newaxis])
        return img_f * vignette_map
