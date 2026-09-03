"""Phase 3 Artistic Style Package."""
from src.art.types import (
    RendererBackend,
    StylePreset,
    StyleConfig,
    ControlMap,
    RenderConfig,
    RenderFrame,
    RenderResult,
)
from src.art.base import ArtisticRenderer
from src.art.control_maps import (
    normalize_points,
    draw_pose_map,
    build_edge_map,
    combine_control_maps,
)
from src.art.style_controller import StyleController
from src.art.preprocess import ControlBuilder
from src.art.temporal import TemporalStabilizer
from src.art.opencv_renderer import OpenCVArtRenderer, OpenCVIllustrationRenderer
from src.art.diffusion_renderer import DiffusionRenderer
from src.art.math_engine import MathematicalStyleEngine
from src.art.mathematical import (
    MathematicalAnimeEngine,
    MathematicalRenderer,
    MathematicalAnimeStyle,
    DEFAULT_ANIME_PALETTE,
)
from src.art.style_engine import StyleEngine
from src.art.video_renderer import VideoRenderer, VideoStyleRenderer

__all__ = [
    "RendererBackend",
    "StylePreset",
    "StyleConfig",
    "ControlMap",
    "RenderConfig",
    "RenderFrame",
    "RenderResult",
    "ArtisticRenderer",
    "normalize_points",
    "draw_pose_map",
    "build_edge_map",
    "combine_control_maps",
    "StyleController",
    "ControlBuilder",
    "TemporalStabilizer",
    "OpenCVArtRenderer",
    "OpenCVIllustrationRenderer",
    "DiffusionRenderer",
    "MathematicalStyleEngine",
    "MathematicalAnimeEngine",
    "MathematicalRenderer",
    "MathematicalAnimeStyle",
    "DEFAULT_ANIME_PALETTE",
    "StyleEngine",
    "VideoRenderer",
    "VideoStyleRenderer",
]
