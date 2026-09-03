"""Mathematical Anime Engine Configuration."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# 7-tier continuous artistic anime palette matching creator lighting and skin tones
DEFAULT_ANIME_PALETTE = np.array([
    [248, 226, 198],  # light skin / highlight
    [224, 176, 137],  # warm skin diffuse
    [192, 137, 100],  # skin shadow
    [166, 118, 105],  # warm midtone / ambient
    [112, 105, 120],  # neutral desk / background shadow
    [72, 66, 75],     # deep cel shadow
    [38, 34, 38],     # ink / dark contour
], dtype=np.float32)


@dataclass(frozen=True)
class MathematicalAnimeStyle:
    """Mathematical specification of the target anime visual style."""

    # Tone
    contrast: float = 1.08
    gamma: float = 0.96
    tone_strength: float = 0.82
    tone_artistic_factor: float = 0.50

    # Color
    saturation: float = 1.08
    palette_mix: float = 0.60
    palette_temperature: float = 0.70

    # Anime simplification
    color_levels: int = 12
    texture_suppression: float = 0.72
    detail_retention: float = 0.28

    # Line art
    edge_strength: float = 0.72
    edge_threshold: float = 0.16
    edge_softness: float = 0.055
    line_darkness: float = 0.82
    ink_color: tuple[int, int, int] = (30, 25, 30)

    # Cel shading
    shadow_threshold: float = 0.40
    shadow_strength: float = 0.20
    shadow_softness: float = 0.06
    highlight_threshold: float = 0.78
    highlight_strength: float = 0.10
    highlight_softness: float = 0.05

    # Face & character
    face_contrast: float = 1.08
    eye_emphasis: float = 1.12
    skin_smoothing: float = 0.70

    # Background
    background_simplification: float = 0.65

    # Cinematic Lighting
    warm_light_strength: float = 0.12
    warm_light_color: tuple[float, float, float] = (1.00, 0.86, 0.68)

    # Temporal consistency
    temporal_strength: float = 0.12
    temporal_motion_limit: float = 0.18
    use_optical_flow: bool = True
