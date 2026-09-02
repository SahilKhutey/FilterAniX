"""Color Grading, Cinematic Lighting, and Anime Skin-Tone Harmony."""
from typing import Optional
import cv2
import numpy as np
from filteranix.core.config import ColorConfig, LightingConfig, CharacterConfig


class ColorGradingEngine:
    """Applies warm studio cinematic lighting, anime skin-tone palette correction, and bloom."""

    def __init__(
        self,
        lighting_config: Optional[LightingConfig] = None,
        color_config: Optional[ColorConfig] = None,
        character_config: Optional[CharacterConfig] = None,
    ):
        self.lighting = lighting_config or LightingConfig()
        self.color = color_config or ColorConfig()
        self.character = character_config or CharacterConfig()

    def process(self, rgb: np.ndarray, person_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies complete color grading and lighting pipeline."""
        img_f = rgb.astype(np.float32)

        # 1. Global Contrast and Saturation
        if self.color.global_contrast != 1.0:
            mean_val = 128.0
            img_f = (img_f - mean_val) * self.color.global_contrast + mean_val
            img_f = np.clip(img_f, 0, 255)

        # Saturation boost via HSV
        if self.color.global_saturation != 1.0:
            hsv = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.color.global_saturation, 0, 255)
            img_f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 2. Cinematic Warmth (Key Light vs Ambient Cool Shadows)
        if self.lighting.cinematic_warmth > 0:
            gray = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            
            # Highlights get warm tint
            key_tint = np.array(self.lighting.key_light_color, dtype=np.float32)
            # Shadows get cool tint
            shadow_tint = np.array(self.lighting.shadow_color, dtype=np.float32)
            
            highlight_weight = (gray ** 1.5)[..., np.newaxis] * self.lighting.cinematic_warmth * 0.25
            shadow_weight = ((1.0 - gray) ** 1.5)[..., np.newaxis] * self.lighting.shadow_coolness * 0.20
            
            img_f = img_f * (1.0 - highlight_weight) + key_tint * highlight_weight
            img_f = img_f * (1.0 - shadow_weight) + shadow_tint * shadow_weight

        # 3. Anime Skin-Tone Protection & Harmonics
        if self.color.skin_tone_protection and person_mask is not None:
            img_f = self._harmonize_skin_tones(img_f, person_mask)

        # 4. Soft Bloom on High-Key Highlights
        if self.lighting.bloom_threshold < 255:
            img_f = self._apply_soft_bloom(img_f)

        return np.clip(img_f, 0, 255).astype(np.uint8)

    def _harmonize_skin_tones(self, img_f: np.ndarray, person_mask: np.ndarray) -> np.ndarray:
        """Detects skin pixels in the person region and gently guides them toward the target anime peach palette."""
        hsv = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Standard human skin hue bounds in OpenCV HSV: H in [0, 25], S in [30, 180], V > 60
        skin_mask = (
            (h >= 0) & (h <= 25) &
            (s >= 25) & (s <= 200) &
            (v >= 60)
        ).astype(np.float32) * person_mask

        skin_mask_soft = cv2.GaussianBlur(skin_mask, (11, 11), 0)[..., np.newaxis]

        target_skin = np.array(self.character.skin_rgb_target, dtype=np.float32)
        # Gently blend skin regions with target peach tone while preserving luminance structure
        luminance = (0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2])[..., np.newaxis] / 128.0
        shaded_target = target_skin * luminance * self.color.skin_warmth_boost

        img_f = img_f * (1.0 - skin_mask_soft * 0.35) + shaded_target * (skin_mask_soft * 0.35)
        return img_f

    def _apply_soft_bloom(self, img_f: np.ndarray) -> np.ndarray:
        """Extracts high-luminance areas and creates a soft anime dreamy glow."""
        gray = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        bright_mask = np.clip((gray.astype(np.float32) - self.lighting.bloom_threshold) / (255 - self.lighting.bloom_threshold + 1e-5), 0.0, 1.0)
        
        bright_pixels = img_f * bright_mask[..., np.newaxis]
        r = max(3, self.lighting.bloom_radius)
        if r % 2 == 0:
            r += 1
        bloom = cv2.GaussianBlur(bright_pixels, (r, r), 0)
        
        return img_f + bloom * 0.30
