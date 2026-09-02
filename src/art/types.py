"""Data Types and Configurations for Phase 3 Artistic Style Engine."""
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np


class RendererBackend(str, Enum):
    OPENCV = "opencv"
    DIFFUSERS = "diffusers"


@dataclass
class StylePreset:
    """Aesthetic parameters defining the visual illustration language."""
    name: str = "creator_anime"
    prompt: str = (
        "clean cinematic anime illustration, young creator character, "
        "expressive eyes, clean line art, warm studio lighting, "
        "detailed but controlled background, soft cel shading, coherent proportions"
    )
    negative_prompt: str = (
        "photorealistic, live action, deformed hands, extra fingers, extra limbs, "
        "duplicate person, malformed face, text, watermark, logo, blurry, noisy, ugly"
    )
    line_weight: float = 1.3
    line_tint: List[int] = field(default_factory=lambda: [30, 24, 36])
    shading_levels: int = 6
    color_warmth: float = 0.65
    key_light_color: List[int] = field(default_factory=lambda: [255, 242, 220])
    shadow_coolness: float = 0.35
    shadow_color: List[int] = field(default_factory=lambda: [65, 70, 95])
    bloom_strength: float = 0.18
    contrast_boost: float = 1.12
    saturation_boost: float = 1.18
    edge_canny_low: int = 50
    edge_canny_high: int = 150


@dataclass
class ControlMap:
    """Multi-channel structural control images extracted from real frame and vision data."""
    edge_map: np.ndarray                        # (H, W) or (H, W, 3) uint8 Canny/XDoG edges
    pose_map: Optional[np.ndarray] = None       # (H, W, 3) uint8 OpenPose-style skeletal lines
    face_mesh_map: Optional[np.ndarray] = None  # (H, W, 3) uint8 Facial contour guides
    hand_map: Optional[np.ndarray] = None       # (H, W, 3) uint8 Hand bone guides
    combined_control: Optional[np.ndarray] = None


@dataclass
class RenderConfig:
    """Master rendering execution configuration."""
    backend: RendererBackend = RendererBackend.OPENCV
    style: StylePreset = field(default_factory=StylePreset)
    temporal_alpha: float = 0.60
    scene_cut_threshold: float = 0.40
    enable_reference_palette: bool = True
    reference_image_path: Optional[str] = None
    device: str = "auto"
    model_id: Optional[str] = None

    # ControlNet structural conditioning
    controlnet_model_id: Optional[str] = None
    controlnet_conditioning_scale: float = 0.80

    # Identity Adapter (IP-Adapter) reference conditioning
    identity_adapter_model_id: Optional[str] = None
    identity_conditioning_scale: float = 0.70
