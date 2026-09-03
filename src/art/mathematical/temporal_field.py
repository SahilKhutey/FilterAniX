from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .temporal_types import TemporalObservation


@dataclass
class TemporalFieldResult:
    """
    Complete output of MTH-10.
    """

    output_rgb: np.ndarray
    current_rgb: np.ndarray
    previous_warped_rgb: np.ndarray

    flow_x: np.ndarray
    flow_y: np.ndarray
    flow_magnitude: np.ndarray

    motion_stability: np.ndarray
    difference_field: np.ndarray
    temporal_confidence: np.ndarray

    temporal_strength: np.ndarray
    valid_warp_field: np.ndarray

    scene_cut: bool


class MathematicalTemporalField:
    """
    MTH-10 — Mathematical Temporal Field Engine.

    Performs deterministic frame-to-frame temporal stabilization.

    The engine never generates a new image independently.
    It transforms the current mathematical frame and optionally
    blends it with a flow-warped copy of the previous mathematical frame.
    """

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:
        self.style = style or MathematicalAnimeStyle.creator_anime()

        self._previous_output: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_style(self) -> None:
        if hasattr(self.style, "validated"):
            self.style.validated()

        strength = float(self.style.temporal_strength)
        motion_limit = float(self.style.temporal_motion_limit)

        if not 0.0 <= strength <= 1.0:
            raise ValueError(
                "temporal_strength must be in [0, 1]"
            )

        if motion_limit <= 0.0:
            raise ValueError(
                "temporal_motion_limit must be > 0"
            )

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")

        if frame.ndim != 3:
            raise ValueError(
                f"frame must have shape HxWx3, got {frame.shape}"
            )

        if frame.shape[2] != 3:
            raise ValueError(
                f"frame must have 3 channels, got {frame.shape}"
            )

        if frame.shape[0] < 2 or frame.shape[1] < 2:
            raise ValueError(
                "frame dimensions must be at least 2x2"
            )

    def _validate_flow(
        self,
        flow: np.ndarray | None,
        height: int,
        width: int,
    ) -> None:
        if flow is None:
            return

        if not isinstance(flow, np.ndarray):
            raise TypeError(
                "optical_flow must be a numpy.ndarray"
            )

        if flow.shape != (height, width, 2):
            raise ValueError(
                "optical_flow must have shape "
                f"({height}, {width}, 2), got {flow.shape}"
            )

        if not np.all(np.isfinite(flow)):
            raise ValueError(
                "optical_flow contains non-finite values"
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, frame: np.ndarray) -> np.ndarray:
        original_dtype = frame.dtype

        frame = frame.astype(
            np.float32,
            copy=False,
        )

        if np.issubdtype(original_dtype, np.integer):
            frame /= 255.0
        elif float(np.max(frame)) > 1.0:
            frame /= 255.0

        return np.clip(
            frame,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Flow
    # ------------------------------------------------------------------

    def calculate_flow_components(
        self,
        optical_flow: np.ndarray | None,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if optical_flow is None:
            return (
                np.zeros((height, width), dtype=np.float32),
                np.zeros((height, width), dtype=np.float32),
            )

        flow = optical_flow.astype(np.float32, copy=False)

        return (
            flow[..., 0],
            flow[..., 1],
        )

    def calculate_flow_magnitude(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
    ) -> np.ndarray:
        magnitude = np.sqrt(
            flow_x * flow_x +
            flow_y * flow_y
        )

        return magnitude.astype(np.float32)

    # ------------------------------------------------------------------
    # Motion stability
    # ------------------------------------------------------------------

    def calculate_motion_stability(
        self,
        flow_magnitude: np.ndarray,
    ) -> np.ndarray:
        limit = max(
            float(self.style.temporal_motion_limit),
            1e-6,
        )

        normalized_motion = flow_magnitude / limit

        normalized_motion = np.clip(
            normalized_motion,
            0.0,
            1.0,
        )

        stability = 1.0 - normalized_motion

        return stability.astype(np.float32)

    # ------------------------------------------------------------------
    # Coordinate system
    # ------------------------------------------------------------------

    def coordinate_grid(
        self,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.arange(
            width,
            dtype=np.float32,
        )

        y = np.arange(
            height,
            dtype=np.float32,
        )

        grid_x, grid_y = np.meshgrid(
            x,
            y,
        )

        return grid_x, grid_y

    # ------------------------------------------------------------------
    # Previous-frame warping
    # ------------------------------------------------------------------

    def warp_previous_frame(
        self,
        previous_rgb: np.ndarray,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = previous_rgb.shape[:2]

        grid_x, grid_y = self.coordinate_grid(
            height,
            width,
        )

        # Optical flow convention:
        #
        # current(x, y) corresponds to
        # previous(x - flow_x, y - flow_y)
        #
        # This creates the sampling coordinates used by remap().
        map_x = grid_x - flow_x
        map_y = grid_y - flow_y

        warped = cv2.remap(
            previous_rgb.astype(np.float32),
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

        valid_x = (
            (map_x >= 0.0)
            & (map_x <= float(width - 1))
        )

        valid_y = (
            (map_y >= 0.0)
            & (map_y <= float(height - 1))
        )

        valid = (
            valid_x & valid_y
        ).astype(np.float32)

        return (
            np.clip(warped, 0.0, 1.0),
            valid,
        )

    # ------------------------------------------------------------------
    # Difference field
    # ------------------------------------------------------------------

    def calculate_difference_field(
        self,
        current_rgb: np.ndarray,
        previous_warped_rgb: np.ndarray,
    ) -> np.ndarray:
        difference = np.mean(
            np.abs(
                current_rgb -
                previous_warped_rgb
            ),
            axis=2,
        )

        return np.clip(
            difference,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Temporal confidence
    # ------------------------------------------------------------------

    def calculate_temporal_confidence(
        self,
        difference_field: np.ndarray,
    ) -> np.ndarray:
        """
        Convert current/previous difference into confidence.

        Small difference:
            confidence -> 1

        Large difference:
            confidence -> 0
        """

        # 0.10 corresponds to approximately 10% normalized RGB error.
        sigma = 0.10

        confidence = np.exp(
            -difference_field / sigma
        )

        return np.clip(
            confidence,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Temporal strength
    # ------------------------------------------------------------------

    def calculate_temporal_strength(
        self,
        motion_stability: np.ndarray,
        temporal_confidence: np.ndarray,
        valid_warp_field: np.ndarray,
        scene_cut: bool,
    ) -> np.ndarray:
        if scene_cut:
            return np.zeros_like(
                motion_stability,
                dtype=np.float32,
            )

        base_strength = float(
            self.style.temporal_strength
        )

        strength = (
            base_strength
            * motion_stability
            * temporal_confidence
            * valid_warp_field
        )

        return np.clip(
            strength,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Temporal blend
    # ------------------------------------------------------------------

    def blend(
        self,
        current_rgb: np.ndarray,
        previous_warped_rgb: np.ndarray,
        temporal_strength: np.ndarray,
    ) -> np.ndarray:
        current_weight = (
            1.0 -
            temporal_strength
        )

        output = (
            current_rgb *
            current_weight[..., None]
            +
            previous_warped_rgb *
            temporal_strength[..., None]
        )

        return np.clip(
            output,
            0.0,
            1.0,
        )

    # ------------------------------------------------------------------
    # Main transformation
    # ------------------------------------------------------------------

    def transform(
        self,
        current_rgb: np.ndarray,
        observation: TemporalObservation | None = None,
    ) -> TemporalFieldResult:
        self._validate_style()
        self._validate_frame(current_rgb)

        current = self._normalize(current_rgb)

        height, width = current.shape[:2]

        if observation is None:
            observation = TemporalObservation()

        self._validate_flow(
            observation.optical_flow,
            height,
            width,
        )

        flow_x, flow_y = (
            self.calculate_flow_components(
                observation.optical_flow,
                height,
                width,
            )
        )

        flow_magnitude = (
            self.calculate_flow_magnitude(
                flow_x,
                flow_y,
            )
        )

        motion_stability = (
            self.calculate_motion_stability(
                flow_magnitude,
            )
        )

        # --------------------------------------------------------------
        # No previous frame = temporal stabilization cannot be applied.
        # --------------------------------------------------------------

        if self._previous_output is None:
            previous_warped = current.copy()

            difference_field = np.zeros(
                (height, width),
                dtype=np.float32,
            )

            temporal_confidence = np.zeros(
                (height, width),
                dtype=np.float32,
            )

            valid_warp = np.zeros(
                (height, width),
                dtype=np.float32,
            )

            temporal_strength = np.zeros(
                (height, width),
                dtype=np.float32,
            )

            output = current.copy()

        else:
            previous = self._previous_output

            if previous.shape != current.shape:
                raise ValueError(
                    "Previous temporal frame resolution does not "
                    "match current frame resolution"
                )

            previous_warped, valid_warp = (
                self.warp_previous_frame(
                    previous,
                    flow_x,
                    flow_y,
                )
            )

            difference_field = (
                self.calculate_difference_field(
                    current,
                    previous_warped,
                )
            )

            temporal_confidence = (
                self.calculate_temporal_confidence(
                    difference_field,
                )
            )

            temporal_strength = (
                self.calculate_temporal_strength(
                    motion_stability,
                    temporal_confidence,
                    valid_warp,
                    observation.scene_cut,
                )
            )

            output = self.blend(
                current,
                previous_warped,
                temporal_strength,
            )

        # --------------------------------------------------------------
        # Scene cuts always force a clean current frame.
        # --------------------------------------------------------------

        if observation.scene_cut:
            output = current.copy()

            temporal_strength = np.zeros(
                (height, width),
                dtype=np.float32,
            )

        # --------------------------------------------------------------
        # Store the FINAL mathematical frame.
        # --------------------------------------------------------------

        self._previous_output = (
            output.astype(
                np.float32,
                copy=True,
            )
        )

        return TemporalFieldResult(
            output_rgb=output,
            current_rgb=current,
            previous_warped_rgb=previous_warped,
            flow_x=flow_x,
            flow_y=flow_y,
            flow_magnitude=flow_magnitude,
            motion_stability=motion_stability,
            difference_field=difference_field,
            temporal_confidence=temporal_confidence,
            temporal_strength=temporal_strength,
            valid_warp_field=valid_warp,
            scene_cut=observation.scene_cut,
        )

    # ------------------------------------------------------------------
    # Public rendering interface
    # ------------------------------------------------------------------

    def render(
        self,
        frame_rgb: np.ndarray,
        observation: TemporalObservation | None = None,
    ) -> np.ndarray:
        result = self.transform(
            frame_rgb,
            observation,
        )

        output = np.clip(
            result.output_rgb * 255.0 + 0.5,
            0.0,
            255.0,
        )

        return output.astype(np.uint8)

    # ------------------------------------------------------------------
    # State control
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset temporal history.

        Must be called when:
        - starting a new video
        - changing resolution
        - seeking
        - restarting playback
        - entering a new sequence
        """

        self._previous_output = None

    @property
    def has_previous_frame(self) -> bool:
        return self._previous_output is not None


# ======================================================================
# Compatibility Class for Older Pipelines
# ======================================================================

class TemporalOpticalFlowField:
    """
    Compatibility wrapper: Maintains inter-frame temporal consistency using optical flow warping.
    """

    def __init__(self, style: MathematicalAnimeStyle):
        self.style = style
        self._prev_source_gray: Optional[np.ndarray] = None
        self._prev_art: Optional[np.ndarray] = None
        self._prev_luminance: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_source_gray = None
        self._prev_art = None
        self._prev_luminance = None

    def stabilize_frame(
        self,
        current_art: np.ndarray,
        current_luminance: np.ndarray,
        current_source_gray: Optional[np.ndarray] = None,
        scene_cut: bool = False,
        precomputed_flow: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        if scene_cut:
            self.reset()

        h, w = current_art.shape[:2]

        if (
            self._prev_art is None
            or self._prev_luminance is None
            or self._prev_luminance.shape != current_luminance.shape
            or self._prev_art.shape != current_art.shape
        ):
            self._prev_art = current_art.copy()
            self._prev_luminance = current_luminance.copy()
            if current_source_gray is not None:
                self._prev_source_gray = current_source_gray.copy()
            return current_art.copy(), 0.0

        motion = float(np.mean(np.abs(current_luminance - self._prev_luminance)))
        limit = max(1e-4, self.style.temporal_motion_limit)
        lambda_t = self.style.temporal_strength * max(0.0, min(1.0, 1.0 - (motion / limit)))

        if lambda_t <= 1e-4:
            self._prev_art = current_art.copy()
            self._prev_luminance = current_luminance.copy()
            if current_source_gray is not None:
                self._prev_source_gray = current_source_gray.copy()
            return current_art.copy(), motion

        warped_prev = self._prev_art
        if getattr(self.style, "use_optical_flow", True):
            flow = None
            if precomputed_flow is not None and precomputed_flow.shape[:2] == (h, w):
                flow = precomputed_flow
            elif current_source_gray is not None and self._prev_source_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(
                        self._prev_source_gray,
                        current_source_gray,
                        None,
                        pyr_scale=0.5,
                        levels=2,
                        winsize=13,
                        iterations=2,
                        poly_n=5,
                        poly_sigma=1.1,
                        flags=0,
                    )
                except Exception:
                    flow = None

            if flow is not None:
                try:
                    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (grid_x - flow[:, :, 0]).astype(np.float32)
                    map_y = (grid_y - flow[:, :, 1]).astype(np.float32)

                    warped_prev = cv2.remap(
                        self._prev_art,
                        map_x,
                        map_y,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT,
                    )
                except Exception:
                    warped_prev = self._prev_art
                warped_prev = self._prev_art

        stabilized = (1.0 - lambda_t) * current_art + lambda_t * warped_prev
        stabilized = np.clip(stabilized, 0.0, 1.0)

        self._prev_art = stabilized.copy()
        self._prev_luminance = current_luminance.copy()
        if current_source_gray is not None:
            self._prev_source_gray = current_source_gray.copy()

        return stabilized, motion
