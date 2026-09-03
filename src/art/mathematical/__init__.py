"""Mathematical Anime Engine Package."""
from .config import DEFAULT_ANIME_PALETTE, MathematicalAnimeStyle
from .color_field import ColorFieldResult, MathematicalColorField, compute_color_field
from .tone_field import ToneFieldResult, MathematicalToneField, compute_tone_field
from .palette_field import PaletteFieldResult, MathematicalPaletteField, compute_palette_projection
from .shadow_field import compute_shadow_field
from .highlight_field import compute_highlight_field
from .shadow_highlight_field import (
    ShadowHighlightFieldResult,
    MathematicalShadowHighlightField,
)
from .edge_field import EdgeFieldResult, MathematicalEdgeField, compute_edge_field
from .geometry_types import (
    GeometryPoint,
    GeometryBox,
    GeometryObservation,
)
from .geometry_field import (
    GeometryFieldResult,
    MathematicalGeometryField,
    compute_surface_normals,
)
from .vision_adapter import adapt_vision_frame
from .face_field import (
    FaceFieldResult,
    MathematicalFaceField,
    compute_face_mask,
    apply_face_modulation,
)
from .texture_field import compute_foreground_mask, apply_background_simplification
from .lighting_field import (
    LightingFieldResult,
    MathematicalLightingField,
    compute_lighting_field,
)
from .temporal_types import TemporalObservation
from .temporal_field import (
    MathematicalTemporalField,
    TemporalFieldResult,
    TemporalOpticalFlowField,
)
from .renderer import (
    MathematicalRenderer,
    MathematicalRenderResult,
)
from .compositor import MathematicalAnimeCompositor
from .diagnostics import MathematicalEngineDiagnostics
from .engine import MathematicalAnimeEngine

__all__ = [
    "DEFAULT_ANIME_PALETTE",
    "MathematicalAnimeStyle",
    "ColorFieldResult",
    "MathematicalColorField",
    "compute_color_field",
    "ToneFieldResult",
    "MathematicalToneField",
    "compute_tone_field",
    "PaletteFieldResult",
    "MathematicalPaletteField",
    "compute_palette_projection",
    "ShadowHighlightFieldResult",
    "MathematicalShadowHighlightField",
    "compute_shadow_field",
    "compute_highlight_field",
    "EdgeFieldResult",
    "MathematicalEdgeField",
    "compute_edge_field",
    "GeometryPoint",
    "GeometryBox",
    "GeometryObservation",
    "GeometryFieldResult",
    "MathematicalGeometryField",
    "FaceFieldResult",
    "MathematicalFaceField",
    "adapt_vision_frame",
    "compute_surface_normals",
    "compute_face_mask",
    "apply_face_modulation",
    "compute_foreground_mask",
    "apply_background_simplification",
    "LightingFieldResult",
    "MathematicalLightingField",
    "compute_lighting_field",
    "TemporalObservation",
    "TemporalFieldResult",
    "MathematicalTemporalField",
    "TemporalOpticalFlowField",
    "MathematicalAnimeCompositor",
    "MathematicalEngineDiagnostics",
    "MathematicalAnimeEngine",
    "MathematicalRenderer",
    "MathematicalRenderResult",
]
