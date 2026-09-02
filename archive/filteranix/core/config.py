"""Configuration system for FilterAniX."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class LineArtConfig:
    algorithm: str = "xdog"
    sigma: float = 0.8
    k_sigma: float = 1.6
    gamma: float = 0.98
    epsilon: float = -0.1
    phi: float = 10.0
    line_weight: float = 1.4
    line_color_tint: List[int] = field(default_factory=lambda: [25, 20, 30])


@dataclass
class CelShadingConfig:
    smoothing_method: str = "kuwahara_bilateral"
    bilateral_diameter: int = 9
    bilateral_sigma_color: float = 60.0
    bilateral_sigma_space: float = 45.0
    kuwahara_radius: int = 4
    quantization_levels: int = 7
    shadow_depth: float = 0.75


@dataclass
class EnvironmentConfig:
    illustration_strength: float = 0.70
    background_blur: float = 0.0
    saturation_boost: float = 1.15
    contrast_boost: float = 1.10
    screentone_enabled: bool = False
    screentone_frequency: float = 35.0
    screentone_angle: float = 45.0


@dataclass
class LightingConfig:
    cinematic_warmth: float = 0.70
    key_light_color: List[int] = field(default_factory=lambda: [255, 240, 215])
    shadow_coolness: float = 0.40
    shadow_color: List[int] = field(default_factory=lambda: [70, 75, 110])
    bloom_threshold: int = 210
    bloom_radius: int = 15


@dataclass
class ColorConfig:
    skin_tone_protection: bool = True
    skin_warmth_boost: float = 1.08
    global_saturation: float = 1.20
    global_contrast: float = 1.12
    monochrome: bool = False


@dataclass
class StyleConfig:
    name: str = "Creator Anime"
    version: str = "1.0"
    line_art: LineArtConfig = field(default_factory=LineArtConfig)
    cel_shading: CelShadingConfig = field(default_factory=CelShadingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    color: ColorConfig = field(default_factory=ColorConfig)


@dataclass
class CharacterConfig:
    character_id: str = "creator_default"
    name: str = "Anime Creator"
    skin_preset: str = "anime_peach"
    skin_rgb_target: List[int] = field(default_factory=lambda: [248, 222, 204])
    hair_accent_color: List[int] = field(default_factory=lambda: [45, 45, 60])
    eye_saturation_boost: float = 1.35
    eye_catchlight_boost: float = 1.25
    clothing_edge_accent: float = 1.25


@dataclass
class VisionConfig:
    enable_segmentation: bool = True
    enable_face_mesh: bool = True
    enable_pose_tracking: bool = True
    enable_hand_tracking: bool = True
    optical_flow_algorithm: str = "dis"
    optical_flow_preset: str = "medium"
    background_accumulation_frames: int = 30


@dataclass
class TemporalConfig:
    enable_temporal_warping: bool = True
    enable_deflicker: bool = True
    warp_blend_alpha: float = 0.65
    temporal_window_size: int = 5
    occlusion_threshold: float = 1.5


@dataclass
class CompositorConfig:
    line_art_blend_mode: str = "multiply"
    line_opacity: float = 0.90
    add_cinematic_vignette: bool = True
    vignette_strength: float = 0.25
    add_soft_bloom: bool = True
    bloom_strength: float = 0.15
    color_grading_strength: float = 0.85


@dataclass
class PipelineConfig:
    target_width: int = 1280
    target_height: int = 720
    target_fps: int = 30
    device: str = "auto"
    vision: VisionConfig = field(default_factory=VisionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    compositor: CompositorConfig = field(default_factory=CompositorConfig)


@dataclass
class FilterAniXConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    character: CharacterConfig = field(default_factory=CharacterConfig)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data or {}


def load_config(
    pipeline_path: Optional[str | Path] = None,
    style_path: Optional[str | Path] = None,
    character_path: Optional[str | Path] = None,
) -> FilterAniXConfig:
    """Loads and merges configuration files into a unified FilterAniXConfig object."""
    config = FilterAniXConfig()

    # Load Pipeline Config
    if pipeline_path:
        p_data = _load_yaml(Path(pipeline_path))
        if "pipeline" in p_data:
            p_sub = p_data["pipeline"]
            config.pipeline.target_width = p_sub.get("target_width", config.pipeline.target_width)
            config.pipeline.target_height = p_sub.get("target_height", config.pipeline.target_height)
            config.pipeline.target_fps = p_sub.get("target_fps", config.pipeline.target_fps)
            config.pipeline.device = p_sub.get("device", config.pipeline.device)
        if "vision" in p_data:
            v_sub = p_data["vision"]
            config.pipeline.vision = VisionConfig(**{k: v for k, v in v_sub.items() if hasattr(VisionConfig, k)})
        if "temporal" in p_data:
            t_sub = p_data["temporal"]
            config.pipeline.temporal = TemporalConfig(**{k: v for k, v in t_sub.items() if hasattr(TemporalConfig, k)})
        if "compositor" in p_data:
            c_sub = p_data["compositor"]
            config.pipeline.compositor = CompositorConfig(**{k: v for k, v in c_sub.items() if hasattr(CompositorConfig, k)})

    # Load Style Config
    if style_path:
        s_data = _load_yaml(Path(style_path))
        if s_data:
            config.style.name = s_data.get("name", config.style.name)
            config.style.version = s_data.get("version", config.style.version)
            if "line_art" in s_data:
                config.style.line_art = LineArtConfig(**{k: v for k, v in s_data["line_art"].items() if hasattr(LineArtConfig, k)})
            if "cel_shading" in s_data:
                config.style.cel_shading = CelShadingConfig(**{k: v for k, v in s_data["cel_shading"].items() if hasattr(CelShadingConfig, k)})
            if "environment" in s_data:
                config.style.environment = EnvironmentConfig(**{k: v for k, v in s_data["environment"].items() if hasattr(EnvironmentConfig, k)})
            if "lighting" in s_data:
                config.style.lighting = LightingConfig(**{k: v for k, v in s_data["lighting"].items() if hasattr(LightingConfig, k)})
            if "color" in s_data:
                config.style.color = ColorConfig(**{k: v for k, v in s_data["color"].items() if hasattr(ColorConfig, k)})

    # Load Character Config
    if character_path:
        c_data = _load_yaml(Path(character_path))
        if c_data:
            config.character.character_id = c_data.get("character_id", config.character.character_id)
            config.character.name = c_data.get("name", config.character.name)
            app = c_data.get("appearance", {})
            config.character.skin_preset = app.get("skin_preset", config.character.skin_preset)
            config.character.skin_rgb_target = app.get("skin_rgb_target", config.character.skin_rgb_target)
            config.character.hair_accent_color = app.get("hair_accent_color", config.character.hair_accent_color)
            eye = app.get("eye_enhancement", {})
            config.character.eye_saturation_boost = eye.get("saturation_boost", config.character.eye_saturation_boost)
            config.character.eye_catchlight_boost = eye.get("catchlight_boost", config.character.eye_catchlight_boost)
            cloth = app.get("clothing", {})
            config.character.clothing_edge_accent = cloth.get("edge_accent_strength", config.character.clothing_edge_accent)

    return config
