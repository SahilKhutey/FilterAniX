from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.art.diffusion_renderer import DiffusionRenderer
from src.art.opencv_renderer import OpenCVIllustrationRenderer
from src.art.types import RenderConfig, StyleConfig, StylePreset
from src.art.preprocess import ControlMap
from src.vision.models import FrameVisionData


@dataclass
class KeyframeRenderResult:
    frame_index: int
    frame: np.ndarray
    backend: str
    used_fallback: bool


class KeyframeRenderer:
    """
    Expensive renderer used ONLY for selected keyframes.

    Backend hierarchy:

        Diffusion
           ↓ failure/unavailable
        OpenCV procedural
    """

    def __init__(
        self,
        config: Optional[RenderConfig | StyleConfig] = None,
    ):

        self.config = (
            config
            or StyleConfig()
        )

        self.diffusion = (
            DiffusionRenderer(
                self.config
            )
        )

        preset = self.config.style if hasattr(self.config, "style") and isinstance(self.config.style, StylePreset) else StylePreset()
        self.fallback = (
            OpenCVIllustrationRenderer(
                preset
            )
        )

    def render(
        self,
        frame_index: int,
        rgb: np.ndarray,
        control_map: ControlMap,
        vision_data: Optional[FrameVisionData],
        reference_rgb: Optional[np.ndarray] = None,
        reference_strength: Optional[float] = None,
    ) -> KeyframeRenderResult:

        # Diffusion backend
        if self.diffusion.pipeline is not None:

            try:

                output = (
                    self.diffusion.render(
                        rgb=rgb,
                        control_map=control_map,
                        vision_data=vision_data,
                        reference_rgb=reference_rgb,
                        reference_strength=reference_strength,
                    )
                )

                return KeyframeRenderResult(
                    frame_index=frame_index,
                    frame=output,
                    backend="diffusion",
                    used_fallback=False,
                )

            except Exception:
                pass

        # Deterministic procedural fallback
        output = self.fallback.render(
            rgb=rgb,
            control_map=control_map,
            vision_data=vision_data,
            reference_rgb=reference_rgb,
        )

        return KeyframeRenderResult(
            frame_index=frame_index,
            frame=output,
            backend="opencv",
            used_fallback=True,
        )
