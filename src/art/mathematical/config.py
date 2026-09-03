from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np


def clamp(value: float, minimum: float, maximum: float) -> float:
    return float(max(minimum, min(maximum, value)))


def validate_rgb(
    value: tuple[int, int, int],
    name: str,
) -> tuple[int, int, int]:

    if len(value) != 3:
        raise ValueError(
            f"{name} must contain exactly 3 RGB channels"
        )

    return tuple(
        max(0, min(255, int(channel)))
        for channel in value
    )


@dataclass(frozen=True)
class MathematicalAnimeStyle:
    """
    Mathematical Anime Style Definition.

    This class contains only deterministic rendering parameters.

    It does not depend on:
        - PyTorch
        - CUDA
        - Diffusers
        - ControlNet
        - IP-Adapter
        - TensorFlow
        - YOLO

    Every field is consumed by a later mathematical rendering stage.
    """

    # ============================================================
    # GLOBAL TONE
    # ============================================================

    contrast: float = 1.08

    gamma: float = 0.96

    tone_strength: float = 0.82


    # ============================================================
    # COLOR FIELD
    # ============================================================

    saturation: float = 1.08

    palette_mix: float = 0.60

    color_levels: int = 12

    palette_temperature: float = 0.70


    # ============================================================
    # LOCAL IMAGE FIELD
    # ============================================================

    smooth_sigma: float = 1.15

    texture_suppression: float = 0.72

    detail_retention: float = 0.28


    # ============================================================
    # ANIME LINE / INK FIELD
    # ============================================================

    edge_strength: float = 0.72

    edge_threshold: float = 0.16

    edge_softness: float = 0.055

    line_darkness: float = 0.82

    edge_gradient_weight: float = 0.65
    edge_laplacian_weight: float = 0.20
    edge_multiscale_weight: float = 0.15

    edge_sigma_small: float = 0.8
    edge_sigma_medium: float = 1.6
    edge_sigma_large: float = 3.0

    edge_percentile: float = 95.0

    line_min_strength: float = 0.05
    line_max_strength: float = 1.0

    line_softness: float = 0.08

    line_preserve_highlights: float = 0.15
    line_preserve_shadows: float = 0.05


    # ============================================================
    # CEL SHADOW FIELD (MTH-06)
    # ============================================================

    shadow_sigma_small: float = 2.0
    shadow_sigma_medium: float = 5.0
    shadow_sigma_large: float = 11.0

    shadow_scale_small: float = 0.20
    shadow_scale_medium: float = 0.35
    shadow_scale_large: float = 0.45

    shadow_threshold: float = 0.40
    shadow_softness: float = 0.08
    shadow_strength: float = 0.20

    shadow_color_mix: float = 0.35
    shadow_temperature: float = 0.25


    # ============================================================
    # HIGHLIGHT FIELD (MTH-06)
    # ============================================================

    highlight_threshold: float = 0.78
    highlight_softness: float = 0.08
    highlight_strength: float = 0.10

    highlight_color_mix: float = 0.25
    highlight_temperature: float = 0.75

    lighting_saturation: float = 1.02
    lighting_global_strength: float = 0.75


    # ============================================================
    # CHARACTER / FACE FIELD
    # ============================================================

    face_contrast: float = 1.08

    face_smoothing: float = 0.70

    eye_emphasis: float = 1.12

    character_detail_retention: float = 0.34

    # MTH-07 Character / Geometry
    geometry_enabled: bool = True

    face_geometry_strength: float = 0.90
    pose_geometry_strength: float = 0.75
    hand_geometry_strength: float = 0.80
    person_geometry_strength: float = 0.70

    geometry_detail_retention: float = 0.85
    geometry_background_simplification: float = 0.65

    face_landmark_sigma: float = 18.0
    pose_landmark_sigma: float = 28.0
    hand_landmark_sigma: float = 16.0

    geometry_field_smoothing: float = 2.0

    # MTH-08 Face / Facial Feature Field
    face_feature_sigma: float = 6.0
    face_detail_retention: float = 0.85
    face_smoothing_strength: float = 0.70
    eye_field_strength: float = 1.12
    mouth_field_strength: float = 1.08
    nose_field_strength: float = 1.05


    # ============================================================
    # BACKGROUND FIELD
    # ============================================================

    background_simplification: float = 0.65

    background_detail_retention: float = 0.18


    # ============================================================
    # LIGHTING FIELD
    # ============================================================

    warm_light_strength: float = 0.18

    warm_light_temperature: float = 0.72


    # ============================================================
    # TEMPORAL FIELD
    # ============================================================

    temporal_strength: float = 0.12

    temporal_motion_limit: float = 0.18

    temporal_scene_cut_reset: bool = True

    use_optical_flow: bool = True


    # ============================================================
    # PROCESSING
    # ============================================================

    working_scale: float = 1.0

    output_dtype: str = "uint8"


    # ============================================================
    # ARTISTIC PALETTE
    #
    # RGB 0..255
    #
    # Designed for the supplied warm creator/anime reference.
    # ============================================================

    palette: tuple[tuple[int, int, int], ...] = field(
        default_factory=lambda: (
            (248, 226, 198),
            (230, 188, 148),
            (202, 150, 112),
            (170, 122, 104),
            (128, 112, 120),
            (88, 78, 90),
            (48, 42, 50),
            (28, 25, 30),
        )
    )


    # ============================================================
    # LINE COLOR
    #
    # Deliberately not pure black.
    # ============================================================

    line_tint: tuple[int, int, int] = (
        32,
        27,
        34,
    )


    # ============================================================
    # WARM KEY LIGHT
    # ============================================================

    key_light_color: tuple[int, int, int] = (
        255,
        239,
        211,
    )


    # ============================================================
    # COOL SHADOW COLOR
    # ============================================================

    shadow_color: tuple[int, int, int] = (
        66,
        70,
        94,
    )


    # ============================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # ============================================================

    @property
    def ink_color(self) -> tuple[int, int, int]:
        return self.line_tint

    @property
    def skin_smoothing(self) -> float:
        return self.face_smoothing

    @property
    def warm_light_color(self) -> tuple[float, float, float]:
        return (
            self.key_light_color[0] / 255.0,
            self.key_light_color[1] / 255.0,
            self.key_light_color[2] / 255.0,
        )

    @property
    def tone_artistic_factor(self) -> float:
        return 0.50


    # ============================================================
    # VALIDATION
    # ============================================================

    def validated(self) -> "MathematicalAnimeStyle":

        palette = tuple(
            validate_rgb(
                color,
                "palette color",
            )
            for color in self.palette
        )

        if len(palette) < 2:
            raise ValueError(
                "Palette must contain at least two colors"
            )

        if self.output_dtype != "uint8":
            raise ValueError(
                "MTH-01 currently supports uint8 output only"
            )

        if self.edge_gradient_weight < 0:
            raise ValueError("edge_gradient_weight must be >= 0")

        if self.edge_laplacian_weight < 0:
            raise ValueError("edge_laplacian_weight must be >= 0")

        if self.edge_multiscale_weight < 0:
            raise ValueError("edge_multiscale_weight must be >= 0")

        if self.edge_sigma_small <= 0:
            raise ValueError("edge_sigma_small must be > 0")

        if self.edge_sigma_medium <= 0:
            raise ValueError("edge_sigma_medium must be > 0")

        if self.edge_sigma_large <= 0:
            raise ValueError("edge_sigma_large must be > 0")

        if not 0 < self.edge_percentile <= 100:
            raise ValueError("edge_percentile must be in (0, 100]")

        if not 0 <= self.line_min_strength <= 1:
            raise ValueError("line_min_strength must be in [0, 1]")

        if not 0 <= self.line_max_strength <= 1:
            raise ValueError("line_max_strength must be in [0, 1]")

        if self.line_min_strength > self.line_max_strength:
            raise ValueError(
                "line_min_strength cannot exceed line_max_strength"
            )

        if not 0.0 <= self.face_geometry_strength <= 1.0:
            raise ValueError(
                "face_geometry_strength must be in [0, 1]"
            )

        if not 0.0 <= self.pose_geometry_strength <= 1.0:
            raise ValueError(
                "pose_geometry_strength must be in [0, 1]"
            )

        if not 0.0 <= self.hand_geometry_strength <= 1.0:
            raise ValueError(
                "hand_geometry_strength must be in [0, 1]"
            )

        if not 0.0 <= self.person_geometry_strength <= 1.0:
            raise ValueError(
                "person_geometry_strength must be in [0, 1]"
            )

        if self.geometry_field_smoothing <= 0:
            raise ValueError(
                "geometry_field_smoothing must be > 0"
            )

        if self.face_feature_sigma <= 0:
            raise ValueError(
                "face_feature_sigma must be > 0"
            )

        if not 0.0 <= self.face_detail_retention <= 1.0:
            raise ValueError(
                "face_detail_retention must be in [0, 1]"
            )

        if not 0.0 <= self.face_smoothing_strength <= 1.0:
            raise ValueError(
                "face_smoothing_strength must be in [0, 1]"
            )

        return MathematicalAnimeStyle(

            # Tone
            contrast=clamp(
                self.contrast,
                0.50,
                1.80,
            ),

            gamma=clamp(
                self.gamma,
                0.50,
                1.50,
            ),

            tone_strength=clamp(
                self.tone_strength,
                0.0,
                1.0,
            ),

            # Color
            saturation=clamp(
                self.saturation,
                0.0,
                2.0,
            ),

            palette_mix=clamp(
                self.palette_mix,
                0.0,
                1.0,
            ),

            color_levels=max(
                2,
                min(64, int(self.color_levels)),
            ),

            palette_temperature=clamp(
                self.palette_temperature,
                0.0,
                1.0,
            ),

            # Local field
            smooth_sigma=clamp(
                self.smooth_sigma,
                0.0,
                4.0,
            ),

            texture_suppression=clamp(
                self.texture_suppression,
                0.0,
                1.0,
            ),

            detail_retention=clamp(
                self.detail_retention,
                0.0,
                1.0,
            ),

            # Line field
            edge_strength=clamp(
                self.edge_strength,
                0.0,
                1.0,
            ),

            edge_threshold=clamp(
                self.edge_threshold,
                0.0,
                1.0,
            ),

            edge_softness=clamp(
                self.edge_softness,
                0.005,
                0.5,
            ),

            line_darkness=clamp(
                self.line_darkness,
                0.0,
                1.0,
            ),

            edge_gradient_weight=float(self.edge_gradient_weight),
            edge_laplacian_weight=float(self.edge_laplacian_weight),
            edge_multiscale_weight=float(self.edge_multiscale_weight),
            edge_sigma_small=float(self.edge_sigma_small),
            edge_sigma_medium=float(self.edge_sigma_medium),
            edge_sigma_large=float(self.edge_sigma_large),
            edge_percentile=float(self.edge_percentile),
            line_min_strength=float(self.line_min_strength),
            line_max_strength=float(self.line_max_strength),
            line_softness=clamp(self.line_softness, 0.001, 1.0),
            line_preserve_highlights=clamp(self.line_preserve_highlights, 0.0, 1.0),
            line_preserve_shadows=clamp(self.line_preserve_shadows, 0.0, 1.0),

            # Shadows (MTH-06)
            shadow_sigma_small=max(0.01, float(self.shadow_sigma_small)),
            shadow_sigma_medium=max(0.01, float(self.shadow_sigma_medium)),
            shadow_sigma_large=max(0.01, float(self.shadow_sigma_large)),
            shadow_scale_small=max(0.0, float(self.shadow_scale_small)),
            shadow_scale_medium=max(0.0, float(self.shadow_scale_medium)),
            shadow_scale_large=max(0.0, float(self.shadow_scale_large)),
            shadow_threshold=clamp(
                self.shadow_threshold,
                0.0,
                1.0,
            ),
            shadow_softness=clamp(
                self.shadow_softness,
                0.005,
                0.5,
            ),
            shadow_strength=clamp(
                self.shadow_strength,
                0.0,
                1.0,
            ),
            shadow_color_mix=clamp(
                self.shadow_color_mix,
                0.0,
                1.0,
            ),
            shadow_temperature=clamp(
                self.shadow_temperature,
                0.0,
                1.0,
            ),

            # Highlights (MTH-06)
            highlight_threshold=clamp(
                self.highlight_threshold,
                0.0,
                1.0,
            ),
            highlight_softness=clamp(
                self.highlight_softness,
                0.005,
                0.5,
            ),
            highlight_strength=clamp(
                self.highlight_strength,
                0.0,
                1.0,
            ),
            highlight_color_mix=clamp(
                self.highlight_color_mix,
                0.0,
                1.0,
            ),
            highlight_temperature=clamp(
                self.highlight_temperature,
                0.0,
                1.0,
            ),
            lighting_saturation=clamp(
                self.lighting_saturation,
                0.0,
                2.0,
            ),
            lighting_global_strength=clamp(
                self.lighting_global_strength,
                0.0,
                1.0,
            ),

            # Character
            face_contrast=clamp(
                self.face_contrast,
                0.5,
                1.8,
            ),

            face_smoothing=clamp(
                self.face_smoothing,
                0.0,
                1.0,
            ),

            eye_emphasis=clamp(
                self.eye_emphasis,
                0.5,
                2.0,
            ),

            character_detail_retention=clamp(
                self.character_detail_retention,
                0.0,
                1.0,
            ),

            # MTH-07 Character / Geometry
            geometry_enabled=bool(self.geometry_enabled),
            face_geometry_strength=clamp(self.face_geometry_strength, 0.0, 1.0),
            pose_geometry_strength=clamp(self.pose_geometry_strength, 0.0, 1.0),
            hand_geometry_strength=clamp(self.hand_geometry_strength, 0.0, 1.0),
            person_geometry_strength=clamp(self.person_geometry_strength, 0.0, 1.0),
            geometry_detail_retention=clamp(self.geometry_detail_retention, 0.0, 1.0),
            geometry_background_simplification=clamp(self.geometry_background_simplification, 0.0, 1.0),
            face_landmark_sigma=max(0.1, float(self.face_landmark_sigma)),
            pose_landmark_sigma=max(0.1, float(self.pose_landmark_sigma)),
            hand_landmark_sigma=max(0.1, float(self.hand_landmark_sigma)),
            geometry_field_smoothing=max(0.01, float(self.geometry_field_smoothing)),

            # MTH-08 Face / Facial Feature Field
            face_feature_sigma=max(0.1, float(self.face_feature_sigma)),
            face_detail_retention=clamp(self.face_detail_retention, 0.0, 1.0),
            face_smoothing_strength=clamp(self.face_smoothing_strength, 0.0, 1.0),
            eye_field_strength=max(0.0, float(self.eye_field_strength)),
            mouth_field_strength=max(0.0, float(self.mouth_field_strength)),
            nose_field_strength=max(0.0, float(self.nose_field_strength)),

            # Background
            background_simplification=clamp(
                self.background_simplification,
                0.0,
                1.0,
            ),

            background_detail_retention=clamp(
                self.background_detail_retention,
                0.0,
                1.0,
            ),

            # Lighting
            warm_light_strength=clamp(
                self.warm_light_strength,
                0.0,
                1.0,
            ),

            warm_light_temperature=clamp(
                self.warm_light_temperature,
                0.0,
                1.0,
            ),

            # Temporal
            temporal_strength=clamp(
                self.temporal_strength,
                0.0,
                0.5,
            ),

            temporal_motion_limit=clamp(
                self.temporal_motion_limit,
                0.001,
                1.0,
            ),

            temporal_scene_cut_reset=bool(
                self.temporal_scene_cut_reset
            ),

            use_optical_flow=bool(
                self.use_optical_flow
            ),

            # Processing
            working_scale=clamp(
                self.working_scale,
                0.25,
                1.0,
            ),

            output_dtype="uint8",

            # Colors
            palette=palette,

            line_tint=validate_rgb(
                self.line_tint,
                "line_tint",
            ),

            key_light_color=validate_rgb(
                self.key_light_color,
                "key_light_color",
            ),

            shadow_color=validate_rgb(
                self.shadow_color,
                "shadow_color",
            ),
        )


    # ============================================================
    # SERIALIZATION
    # ============================================================

    def to_dict(self) -> dict[str, Any]:

        return asdict(
            self.validated()
        )


    # ============================================================
    # DEFAULT TARGET STYLE
    # ============================================================

    @classmethod
    def creator_anime(
        cls,
    ) -> "MathematicalAnimeStyle":

        return cls().validated()


# Continuous artistic anime palette as numpy float32 array
DEFAULT_ANIME_PALETTE = np.array(
    MathematicalAnimeStyle().palette,
    dtype=np.float32,
)
