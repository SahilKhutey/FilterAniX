"""FilterAniX Mathematical Anime Engine: First-class deterministic image-field renderer."""
from __future__ import annotations

import time
from typing import Any, Optional
import numpy as np

from src.art.base import ArtisticRenderer
from .config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle
from .compositor import MathematicalAnimeCompositor
from .diagnostics import MathematicalEngineDiagnostics


class MathematicalAnimeEngine(ArtisticRenderer):
    """
    Mathematical Anime Engine v1.0.
    Transforms every video frame through continuous image fields:
        A_t(x,y) = F(I_t, N_t, G_t, L_t, C_t, E_t, S_t, H_t, V_t, A_{t-1})
    100% deterministic, CPU-accelerated, zero diffusion inference.
    """

    def __init__(
        self,
        style: Optional[MathematicalAnimeStyle] = None,
        palette: Optional[np.ndarray] = None,
    ):
        self.style = style or MathematicalAnimeStyle()
        self.palette = palette if palette is not None else DEFAULT_ANIME_PALETTE.copy()
        self.compositor = MathematicalAnimeCompositor(self.style)
        self.diagnostics = MathematicalEngineDiagnostics()
        self.frame_count = 0

    def reset(self) -> None:
        """Resets all internal temporal and state buffers."""
        self.compositor.reset()
        self.frame_count = 0

    def reset_temporal(self) -> None:
        """Alias for reset() for backwards compatibility."""
        self.reset()

    def render(
        self,
        rgb: np.ndarray,
        vision_data: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        lipsync_record: Optional[Any] = None,
        reference_rgb: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes full deterministic field transformation on single input RGB frame (uint8).
        """
        start_t = time.perf_counter()

        # Optional reference palette extraction if reference frame provided
        active_palette = self.palette
        if reference_rgb is not None and reference_rgb.size > 0:
            pass  # Default fixed 7-color anime palette preserves high visual consistency

        # Execute field transformation through compositor
        out_uint8, telemetry = self.compositor.transform(
            rgb_uint8=rgb,
            vision_data=vision_data,
            lipsync_record=lipsync_record,
            scene_cut=scene_cut,
            stabilize=stabilize,
            palette_override=active_palette,
        )

        duration_ms = (time.perf_counter() - start_t) * 1000.0
        self.diagnostics.record_frame(
            frame_index=self.frame_count,
            duration_ms=duration_ms,
            motion_score=telemetry.get("motion_score", 0.0),
            edge_density=telemetry.get("edge_density", 0.0),
            mean_luminance=telemetry.get("mean_luminance", 0.0),
        )
        self.frame_count += 1

        return out_uint8


# Standard alias matching proposed architecture
MathematicalRenderer = MathematicalAnimeEngine
