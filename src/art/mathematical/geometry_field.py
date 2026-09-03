from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .geometry_types import (
    GeometryBox,
    GeometryObservation,
    GeometryPoint,
)


@dataclass
class GeometryFieldResult:
    output_rgb: np.ndarray
    input_rgb: np.ndarray

    face_field: np.ndarray
    face_landmark_field: np.ndarray

    pose_field: np.ndarray
    hand_field: np.ndarray

    person_field: np.ndarray
    character_field: np.ndarray
    background_field: np.ndarray

    face_importance: np.ndarray
    structural_importance: np.ndarray

    detail_preservation: np.ndarray
    simplification_field: np.ndarray


class MathematicalGeometryField:
    """
    MTH-07 Mathematical Character / Geometry Field.

    Converts Phase-2 vision observations into continuous
    spatial fields used to control mathematical image
    transformation.

    This engine does not perform identity recognition.

    It measures spatial/structural importance only.
    """

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:

        self.style = (
            style
            or MathematicalAnimeStyle.creator_anime()
        )

        if hasattr(
            self.style,
            "validated",
        ):
            self.style = self.style.validated()

    # =========================================================
    # Validation
    # =========================================================

    def _validate_frame(
        self,
        frame: np.ndarray,
    ) -> None:

        if not isinstance(
            frame,
            np.ndarray,
        ):
            raise TypeError(
                "frame must be a numpy.ndarray"
            )

        if frame.ndim != 3:
            raise ValueError(
                "frame must have shape HxWx3"
            )

        if frame.shape[2] != 3:
            raise ValueError(
                "frame must have exactly 3 channels"
            )

    def _validate_observation(
        self,
        observation: GeometryObservation,
        width: int,
        height: int,
    ) -> None:

        if not isinstance(
            observation,
            GeometryObservation,
        ):
            raise TypeError(
                "observation must be GeometryObservation"
            )

        if (
            observation.width != width
            or observation.height != height
        ):
            raise ValueError(
                "Geometry observation resolution "
                "does not match frame resolution"
            )

    # =========================================================
    # Coordinate grid
    # =========================================================

    def coordinate_grid(
        self,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        y, x = np.mgrid[
            0:height,
            0:width,
        ]

        return (
            x.astype(np.float32),
            y.astype(np.float32),
        )

    # =========================================================
    # Bounding box field
    # =========================================================

    def bounding_box_field(
        self,
        box: GeometryBox | None,
        height: int,
        width: int,
        softness: float = 8.0,
    ) -> np.ndarray:

        if box is None:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        x, y = self.coordinate_grid(
            height,
            width,
        )

        x0 = float(
            min(box.x0, box.x1)
        )

        x1 = float(
            max(box.x0, box.x1)
        )

        y0 = float(
            min(box.y0, box.y1)
        )

        y1 = float(
            max(box.y0, box.y1)
        )

        dx = np.maximum(
            np.maximum(
                x0 - x,
                0.0,
            ),
            x - x1,
        )

        dy = np.maximum(
            np.maximum(
                y0 - y,
                0.0,
            ),
            y - y1,
        )

        distance = np.sqrt(
            dx * dx + dy * dy
        )

        field = np.exp(
            -distance
            / max(
                softness,
                1e-5,
            )
        )

        field *= np.clip(
            box.confidence,
            0.0,
            1.0,
        )

        return np.clip(
            field,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Landmark field
    # =========================================================

    def landmark_field(
        self,
        landmarks: list[GeometryPoint] | None,
        height: int,
        width: int,
        sigma: float,
    ) -> np.ndarray:

        if not landmarks:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        field = np.zeros(
            (height, width),
            dtype=np.float32,
        )

        x, y = self.coordinate_grid(
            height,
            width,
        )

        sigma = max(
            float(sigma),
            1e-5,
        )

        denominator = (
            2.0
            * sigma
            * sigma
        )

        for point in landmarks:

            px = float(point.x)
            py = float(point.y)

            distance_sq = (
                (x - px) ** 2
                + (y - py) ** 2
            )

            influence = np.exp(
                -distance_sq
                / denominator
            )

            influence *= np.clip(
                point.confidence,
                0.0,
                1.0,
            )

            field = np.maximum(
                field,
                influence,
            )

        return np.clip(
            field,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Person segmentation
    # =========================================================

    def person_mask_field(
        self,
        mask: np.ndarray | None,
        height: int,
        width: int,
    ) -> np.ndarray:

        if mask is None:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        if mask.ndim != 2:
            raise ValueError(
                "person_mask must be HxW"
            )

        if (
            mask.shape[0] != height
            or mask.shape[1] != width
        ):
            raise ValueError(
                "person_mask resolution does not "
                "match frame"
            )

        field = mask.astype(
            np.float32
        )

        if field.max() > 1.0:
            field /= 255.0

        return np.clip(
            field,
            0.0,
            1.0,
        )

    # =========================================================
    # Gaussian smoothing
    # =========================================================

    def smooth_field(
        self,
        field: np.ndarray,
        sigma: float = 3.0,
    ) -> np.ndarray:

        sigma = max(
            float(sigma),
            0.01,
        )

        kernel = max(
            3,
            int(
                round(
                    sigma * 6.0 + 1
                )
            ),
        )

        if kernel % 2 == 0:
            kernel += 1

        return cv2.GaussianBlur(
            field,
            (
                kernel,
                kernel,
            ),
            sigmaX=sigma,
            sigmaY=sigma,
        ).astype(np.float32)

    # =========================================================
    # Character field
    # =========================================================

    def calculate_character_field(
        self,
        face_field: np.ndarray,
        pose_field: np.ndarray,
        hand_field: np.ndarray,
        person_field: np.ndarray,
    ) -> np.ndarray:

        combined = np.maximum(
            face_field,
            pose_field,
        )

        combined = np.maximum(
            combined,
            hand_field,
        )

        combined = np.maximum(
            combined,
            person_field,
        )

        return self.smooth_field(
            np.clip(
                combined,
                0.0,
                1.0,
            ),
            sigma=2.0,
        )

    # =========================================================
    # Face importance
    # =========================================================

    def calculate_face_importance(
        self,
        face_field: np.ndarray,
        face_landmark_field: np.ndarray,
    ) -> np.ndarray:

        importance = (
            0.65 * face_field
            + 0.35 * face_landmark_field
        )

        return np.clip(
            importance,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Structural importance
    # =========================================================

    def calculate_structural_importance(
        self,
        character_field: np.ndarray,
        face_importance: np.ndarray,
    ) -> np.ndarray:

        structural = (
            0.70 * character_field
            + 0.30 * face_importance
        )

        return np.clip(
            structural,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Detail preservation
    # =========================================================

    def calculate_detail_preservation(
        self,
        structural_importance: np.ndarray,
    ) -> np.ndarray:

        base = float(
            getattr(
                self.style,
                "character_detail_retention",
                0.34,
            )
        )

        result = (
            base
            + (
                1.0 - base
            )
            * structural_importance
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Background simplification
    # =========================================================

    def calculate_simplification(
        self,
        character_field: np.ndarray,
    ) -> np.ndarray:

        background = 1.0 - character_field

        strength = float(
            getattr(
                self.style,
                "background_simplification",
                0.65,
            )
        )

        result = (
            background
            * strength
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Apply geometry field
    # =========================================================

    def apply_geometry(
        self,
        rgb: np.ndarray,
        detail_preservation: np.ndarray,
        simplification_field: np.ndarray,
    ) -> np.ndarray:

        # Preserve character structure.
        character_detail = (
            0.90
            + 0.10
            * detail_preservation
        )

        result = (
            rgb
            * character_detail[..., None]
        )

        # Suppress only a small amount of background
        # micro-variation. MTH-07 does not destroy the
        # background; it prepares it for later artistic
        # simplification.
        background_factor = (
            1.0
            - 0.08
            * simplification_field
        )

        result *= (
            background_factor[..., None]
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # =========================================================
    # Transform
    # =========================================================

    def transform(
        self,
        frame_rgb: np.ndarray,
        observation: GeometryObservation,
    ) -> GeometryFieldResult:

        self._validate_frame(
            frame_rgb
        )

        height, width = (
            frame_rgb.shape[:2]
        )

        self._validate_observation(
            observation,
            width,
            height,
        )

        original_dtype = (
            frame_rgb.dtype
        )

        rgb = frame_rgb.astype(
            np.float32
        )

        if np.issubdtype(
            original_dtype,
            np.integer,
        ):
            rgb /= 255.0

        elif rgb.max() > 1.0:
            rgb /= 255.0

        rgb = np.clip(
            rgb,
            0.0,
            1.0,
        )

        # -----------------------------------------------------
        # Face
        # -----------------------------------------------------

        face_field = (
            self.bounding_box_field(
                observation.face_box,
                height,
                width,
                softness=10.0,
            )
        )

        face_landmark_field = (
            self.landmark_field(
                observation.face_landmarks,
                height,
                width,
                sigma=18.0,
            )
        )

        # -----------------------------------------------------
        # Pose
        # -----------------------------------------------------

        pose_field = (
            self.landmark_field(
                observation.pose_landmarks,
                height,
                width,
                sigma=28.0,
            )
        )

        # -----------------------------------------------------
        # Hands
        # -----------------------------------------------------

        hand_field = (
            self.landmark_field(
                observation.hand_landmarks,
                height,
                width,
                sigma=16.0,
            )
        )

        # -----------------------------------------------------
        # Person
        # -----------------------------------------------------

        person_field = (
            self.person_mask_field(
                observation.person_mask,
                height,
                width,
            )
        )

        # -----------------------------------------------------
        # Combined character geometry
        # -----------------------------------------------------

        character_field = (
            self.calculate_character_field(
                face_field,
                pose_field,
                hand_field,
                person_field,
            )
        )

        background_field = np.clip(
            1.0 - character_field,
            0.0,
            1.0,
        ).astype(np.float32)

        # -----------------------------------------------------
        # Importance
        # -----------------------------------------------------

        face_importance = (
            self.calculate_face_importance(
                face_field,
                face_landmark_field,
            )
        )

        structural_importance = (
            self.calculate_structural_importance(
                character_field,
                face_importance,
            )
        )

        detail_preservation = (
            self.calculate_detail_preservation(
                structural_importance,
            )
        )

        simplification_field = (
            self.calculate_simplification(
                character_field,
            )
        )

        # -----------------------------------------------------
        # Apply
        # -----------------------------------------------------

        output = self.apply_geometry(
            rgb,
            detail_preservation,
            simplification_field,
        )

        output_rgb = np.clip(
            output * 255.0,
            0.0,
            255.0,
        ).round().astype(
            np.uint8
        )

        return GeometryFieldResult(
            output_rgb=output_rgb,

            input_rgb=np.clip(
                rgb * 255.0,
                0.0,
                255.0,
            ).round().astype(
                np.uint8
            ),

            face_field=face_field,
            face_landmark_field=face_landmark_field,

            pose_field=pose_field,
            hand_field=hand_field,

            person_field=person_field,
            character_field=character_field,
            background_field=background_field,

            face_importance=face_importance,
            structural_importance=structural_importance,

            detail_preservation=detail_preservation,
            simplification_field=simplification_field,
        )

    def render(
        self,
        frame_rgb: np.ndarray,
        observation: GeometryObservation,
    ) -> np.ndarray:

        return self.transform(
            frame_rgb,
            observation,
        ).output_rgb


def compute_surface_normals(
    luminance: np.ndarray,
    depth_scale: float = 2.0,
) -> np.ndarray:
    """
    Compatibility function: Approximates 3D surface normal vector field from luminance gradients.
    """
    Gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)

    nx = -Gx * depth_scale
    ny = -Gy * depth_scale
    nz = np.ones_like(luminance, dtype=np.float32)

    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    norm = np.maximum(norm, 1e-6)

    normals = np.stack([nx / norm, ny / norm, nz / norm], axis=-1)
    return normals
