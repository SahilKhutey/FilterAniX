from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RendererBackend(str, Enum):
    MATHEMATICAL = "mathematical"
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
    line_tint: list[int] = field(default_factory=lambda: [30, 24, 36])
    shading_levels: int = 6
    color_warmth: float = 0.65
    key_light_color: list[int] = field(default_factory=lambda: [255, 242, 220])
    shadow_coolness: float = 0.35
    shadow_color: list[int] = field(default_factory=lambda: [65, 70, 95])
    bloom_strength: float = 0.18
    contrast_boost: float = 1.12
    saturation_boost: float = 1.18
    edge_canny_low: int = 50
    edge_canny_high: int = 150


@dataclass
class StyleConfig:
    name: str = "anime_creator"

    # General image-to-image controls
    denoise_strength: float = 0.35
    guidance_scale: float = 6.5
    inference_steps: int = 20

    # ControlNet
    use_controlnet: bool = True
    controlnet_model_id: str | None = None
    controlnet_conditioning_scale: float = 0.80

    # IP-Adapter
    use_ip_adapter: bool = True
    ip_adapter_repo_id: str = "h94/IP-Adapter"
    ip_adapter_subfolder: str = "models"
    ip_adapter_weight_name: str = "ip-adapter_sd15.bin"
    ip_adapter_scale: float = 0.65
    identity_adapter_model_id: str | None = None
    identity_conditioning_scale: float = 0.70

    # P2 Identity Consistency
    identity_enabled: bool = True
    identity_warning_threshold: float = 0.62
    identity_severe_threshold: float = 0.48
    identity_max_retries: int = 2
    identity_reference_bank_size: int = 8
    identity_evaluation_interval: int = 1
    identity_reference_refresh_score: float = 0.78

    # Temporal behavior
    temporal_blend: float = 0.16
    temporal_alpha: float = 0.60
    keyframe_interval: int = 12
    scene_cut_threshold: float = 0.40

    # Edge preprocessing
    canny_low: int = 80
    canny_high: int = 160

    positive_prompt: str = (
        "high quality animated character, clean line art, "
        "expressive face, polished illustration, consistent character"
    )

    negative_prompt: str = (
        "photorealistic, realistic skin texture, blurry, distorted face, "
        "extra fingers, extra limbs, deformed hands, duplicate person"
    )

    # Mathematical Style Engine Parameters
    smooth_sigma: float = 1.15
    tone_strength: float = 0.82
    tone_contrast: float = 1.08
    tone_gamma: float = 0.96
    color_saturation: float = 1.12
    color_palette_mix: float = 0.58
    color_levels: int = 12
    edge_strength: float = 0.72
    edge_threshold: float = 0.16
    edge_softness: float = 0.055
    shadow_threshold: float = 0.40
    shadow_strength: float = 0.20
    highlight_threshold: float = 0.78
    highlight_strength: float = 0.10
    detail_strength: float = 0.18
    motion_limit: float = 0.18

    # General options
    backend: RendererBackend = RendererBackend.MATHEMATICAL
    style: StylePreset = field(default_factory=StylePreset)
    enable_reference_palette: bool = True
    reference_image_path: str | None = None
    device: str = "auto"
    model_id: str | None = None


@dataclass
class ControlMap:
    combined_control: Any = None
    edge_map: Any | None = None
    pose_map: Any | None = None
    face_map: Any | None = None
    face_mesh_map: Any | None = None
    hand_map: Any | None = None


@dataclass
class RenderFrame:
    frame_index: int
    timestamp: float

    source_frame: Any

    control_map: ControlMap | None = None

    reference_frame: Any | None = None

    scene_id: int = 0
    scene_cut: bool = False
    keyframe: bool = False

    motion_score: float = 0.0
    reference_strength: float = 0.55

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderResult:
    frame_index: int
    frame: Any
    backend: str

    scene_id: int
    keyframe: bool

    metadata: dict[str, Any] = field(default_factory=dict)


# Backwards compatibility alias
RenderConfig = StyleConfig
