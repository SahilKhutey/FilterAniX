"""Deterministic Mathematical Video-to-Style Transformation Engine."""
from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from src.art.base import ArtisticRenderer
from src.art.types import StyleConfig
from src.art.mathematical.config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle
from src.art.mathematical.engine import MathematicalAnimeEngine


class MathematicalStyleEngine(ArtisticRenderer):
    """
    Deterministic Mathematical Video-to-Style Transformation Engine.
    Transforms every input pixel through composable mathematical fields:
        A_t(x,y) = F(I_t, N_t, G_t, L_t, C_t, E_t, S_t, H_t, V_t, A_{t-1})
    with continuous temporal regularization and optical flow stabilization.
    """

    def __init__(self, config: Optional[Union[StyleConfig, MathematicalAnimeStyle]] = None):
        if isinstance(config, MathematicalAnimeStyle):
            self.math_style = config
            self.config = StyleConfig()
        elif isinstance(config, StyleConfig):
            self.config = config
            self.math_style = config.to_mathematical_style()
        else:
            self.config = StyleConfig()
            self.math_style = self.config.to_mathematical_style()

        self.palette = DEFAULT_ANIME_PALETTE.copy()
        self._engine = MathematicalAnimeEngine(self.math_style, palette=self.palette)

    def reset(self) -> None:
        """Resets temporal and state memory across scene cuts or hard boundaries."""
        self._engine.reset()

    def reset_temporal(self) -> None:
        """Resets temporal memory across scene cuts or hard boundaries."""
        self.reset()

    def render(
        self,
        rgb: np.ndarray,
        vision_data: Optional[Any] = None,
        lipsync_record: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        reference_rgb: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes full deterministic mathematical transformation on a single frame.
        Returns uint8 RGB array with identical spatial dimensions (H x W x 3).
        """
        return self._engine.render(
            rgb=rgb,
            vision_data=vision_data,
            lipsync_record=lipsync_record,
            scene_cut=scene_cut,
            stabilize=stabilize,
            reference_rgb=reference_rgb,
            **kwargs,
        )
