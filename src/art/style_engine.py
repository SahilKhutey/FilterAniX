"""Master Style Engine Orchestrator."""
from typing import Optional
import numpy as np

from src.art.types import RenderConfig, RendererBackend
from src.art.preprocess import ControlBuilder
from src.art.temporal import TemporalStabilizer
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.art.diffusion_renderer import DiffusionRenderer
from src.vision.models import FrameVisionData


class StyleEngine:
    """Orchestrates structural control extraction, pluggable rendering backends, and temporal stabilization."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.control_builder = ControlBuilder(self.config.style)
        self.temporal_stabilizer = TemporalStabilizer(
            alpha=self.config.temporal_alpha,
            scene_cut_threshold=self.config.scene_cut_threshold,
        )

        if self.config.backend == RendererBackend.DIFFUSERS:
            self.renderer = DiffusionRenderer(self.config)
        else:
            self.renderer = OpenCVIllustrationRenderer(self.config.style)

    def render_frame(
        self,
        rgb: np.ndarray,
        vision_data: Optional[FrameVisionData] = None,
        reference_rgb: Optional[np.ndarray] = None,
        stabilize: bool = True,
    ) -> np.ndarray:
        """Processes a single raw RGB frame into a stylized anime illustration with temporal stability."""
        # 1. Build structural control map (edges, pose, face, hands)
        control_map = self.control_builder.build_control_map(rgb, vision_data)

        # 2. Render stylized frame
        art_frame = self.renderer.render(
            rgb=rgb,
            control_map=control_map,
            vision_data=vision_data,
            reference_rgb=reference_rgb,
        )

        # 3. Apply temporal stabilization & scene-cut handling
        if stabilize:
            final_frame = self.temporal_stabilizer.stabilize(rgb, art_frame)
        else:
            final_frame = art_frame

        return final_frame

    def reset_temporal(self):
        """Resets temporal stabilizer history."""
        self.temporal_stabilizer.reset()
