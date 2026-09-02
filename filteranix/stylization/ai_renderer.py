"""Unified Stylization Renderer."""
from typing import Optional
import numpy as np
from filteranix.core.config import StyleConfig, CharacterConfig
from filteranix.stylization.cel_shading import CelShader
from filteranix.stylization.line_art import LineArtExtractor
from filteranix.stylization.color_grading import ColorGradingEngine


class StyleRenderer:
    """Orchestrates anime line-art extraction, cel-shading, and color grading for a frame or layer."""

    def __init__(self, style_config: StyleConfig, character_config: CharacterConfig):
        self.style_config = style_config
        self.character_config = character_config
        
        self.line_extractor = LineArtExtractor(style_config.line_art)
        self.cel_shader = CelShader(style_config.cel_shading)
        self.color_grader = ColorGradingEngine(
            lighting_config=style_config.lighting,
            color_config=style_config.color,
            character_config=character_config,
        )

    def render_layer(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
        is_foreground: bool = True,
    ) -> np.ndarray:
        """Transforms an RGB layer (person or background) into stylized anime art."""
        # Step 1: Edge-preserving smoothing & cel-quantization
        cel = self.cel_shader.apply_cel_shading(rgb)

        # Step 2: Color grading, cinematic tone mapping, skin protection
        graded = self.color_grader.process(cel, person_mask=mask if is_foreground else None)

        return graded

    def extract_lines(self, rgb: np.ndarray, depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """Extracts stylized ink lines."""
        return self.line_extractor.extract_xdog(rgb, depth_map=depth_map)
