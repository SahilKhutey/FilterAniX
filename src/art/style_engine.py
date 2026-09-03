"""Master Style Engine Orchestrator."""
from __future__ import annotations

from typing import Optional, Any
import numpy as np

from src.art.types import RenderConfig, RendererBackend
from src.art.preprocess import ControlBuilder
from src.art.temporal import TemporalStabilizer
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.art.diffusion_renderer import DiffusionRenderer
from src.art.math_engine import MathematicalStyleEngine
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
        elif self.config.backend == RendererBackend.MATHEMATICAL:
            self.renderer = MathematicalStyleEngine(self.config)
        else:
            self.renderer = OpenCVIllustrationRenderer(self.config.style)

    def render_frame(
        self,
        rgb: np.ndarray,
        vision_data: Optional[FrameVisionData] = None,
        reference_rgb: Optional[np.ndarray] = None,
        stabilize: bool = True,
        lipsync_record: Optional[Any] = None,
        temporal_decision: Optional[Any] = None,
    ) -> np.ndarray:
        """Processes a single raw RGB frame into a stylized anime illustration with temporal stability and lip-sync."""
        is_cut = False
        if temporal_decision is not None:
            if getattr(temporal_decision, "is_scene_cut", False):
                is_cut = True
                self.reset_temporal()

        # 1. Direct Mathematical Engine dispatch (processes every pixel deterministically)
        if isinstance(self.renderer, MathematicalStyleEngine):
            return self.renderer.render(
                rgb=rgb,
                vision_data=vision_data,
                lipsync_record=lipsync_record,
                scene_cut=is_cut,
                stabilize=stabilize,
                reference_rgb=reference_rgb,
            )

        # 2. Procedural / Generative Fallbacks
        control_map = self.control_builder.build_control_map(rgb, vision_data)

        if hasattr(self.renderer, "render"):
            try:
                art_frame = self.renderer.render(
                    rgb=rgb,
                    control_map=control_map,
                    vision_data=vision_data,
                    reference_rgb=reference_rgb,
                    lipsync_record=lipsync_record,
                )
            except TypeError:
                art_frame = self.renderer.render(
                    rgb=rgb,
                    control_map=control_map,
                    vision_data=vision_data,
                    reference_rgb=reference_rgb,
                )
        else:
            art_frame = rgb

        should_stabilize = stabilize
        if temporal_decision is not None and not getattr(temporal_decision, "preserve_previous", True):
            should_stabilize = False

        if should_stabilize:
            final_frame = self.temporal_stabilizer.stabilize(rgb, art_frame)
        else:
            final_frame = art_frame

        return final_frame

    def reset_temporal(self):
        """Resets temporal stabilizer history."""
        if hasattr(self.renderer, "reset_temporal"):
            self.renderer.reset_temporal()
        self.temporal_stabilizer.reset()
