"""Anchor Keyframe and Static Canvas Manager."""
from typing import Optional
import numpy as np
from filteranix.core.config import FilterAniXConfig
from filteranix.stylization.ai_renderer import StyleRenderer


class AnchorManager:
    """Maintains static anchor canvases (e.g. 100% stable room background plate) across long video sequences."""

    def __init__(self, style_renderer: StyleRenderer, config: FilterAniXConfig):
        self.style_renderer = style_renderer
        self.config = config
        self.stylized_bg_plate: Optional[np.ndarray] = None
        self.stylized_bg_lines: Optional[np.ndarray] = None

    def initialize_static_background(self, raw_bg_plate: np.ndarray):
        """Pre-renders and locks the entire background scene as a persistent stylized illustrated canvas."""
        # Render stylized illustrated background once
        self.stylized_bg_plate = self.style_renderer.render_layer(
            raw_bg_plate, mask=None, depth_map=None, is_foreground=False
        )
        # Extract crisp illustrated background lines once
        self.stylized_bg_lines = self.style_renderer.extract_lines(raw_bg_plate)

    def get_stylized_background(self) -> Optional[np.ndarray]:
        return self.stylized_bg_plate

    def get_stylized_bg_lines(self) -> Optional[np.ndarray]:
        return self.stylized_bg_lines
