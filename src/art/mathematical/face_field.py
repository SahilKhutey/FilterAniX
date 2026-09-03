from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .geometry_types import GeometryObservation


@dataclass
class FaceFieldResult:
    output_rgb: np.ndarray
    input_rgb: np.ndarray

    face_field: np.ndarray
    eye_field: np.ndarray
    nose_field: np.ndarray
    mouth_field: np.ndarray

    central_feature_field: np.ndarray
    facial_geometry_field: np.ndarray

    face_importance: np.ndarray
    detail_preservation: np.ndarray
    smoothing_field: np.ndarray

    eye_emphasis: np.ndarray
    mouth_emphasis: np.ndarray
    nose_emphasis: np.ndarray


class MathematicalFaceField:
    """
    MTH-08

    Mathematical facial feature field engine.

    This module does not generate or replace faces.

    It creates continuous mathematical control fields for:
        - face region
        - eyes
        - nose
        - mouth
        - facial importance
        - detail preservation
        - smoothing
        - feature emphasis

    Input:
        RGB frame + Phase-2 / MTH-07 geometry observation

    Output:
        RGB frame + facial control fields
    """

    # MediaPipe Face Mesh landmark indices.
    LEFT_EYE = 33
    RIGHT_EYE = 263

    NOSE = 1
    NOSE_BRIDGE = 168

    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_UPPER = 13
    MOUTH_LOWER = 14

    CHIN = 152
    FOREHEAD = 10

    def __init__(
        self,
        style: MathematicalAnimeStyle | None = None,
    ) -> None:
        self.style = style or MathematicalAnimeStyle.creator_anime()

        self._validate_style()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_style(self) -> None:
        required = (
            "face_contrast",
            "face_smoothing",
            "eye_emphasis",
            "character_detail_retention",
        )

        for name in required:
            if not hasattr(self.style, name):
                raise AttributeError(
                    f"MathematicalAnimeStyle is missing required field: {name}"
                )

    def _validate_frame(self, frame_rgb: np.ndarray) -> None:
        if not isinstance(frame_rgb, np.ndarray):
            raise TypeError("frame_rgb must be a numpy.ndarray")

        if frame_rgb.ndim != 3:
            raise ValueError("frame_rgb must have shape (H, W, 3)")

        if frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb must have exactly 3 channels")

        if frame_rgb.shape[0] < 2 or frame_rgb.shape[1] < 2:
            raise ValueError("frame_rgb resolution is too small")

    def _validate_observation(
        self,
        observation: GeometryObservation,
        frame_rgb: np.ndarray,
    ) -> None:
        if observation.width != frame_rgb.shape[1]:
            raise ValueError(
                "Geometry observation width does not match frame width"
            )

        if observation.height != frame_rgb.shape[0]:
            raise ValueError(
                "Geometry observation height does not match frame height"
            )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb.dtype == np.uint8:
            return frame_rgb.astype(np.float32) / 255.0

        result = frame_rgb.astype(np.float32)

        if result.max(initial=0.0) > 1.0:
            result /= 255.0

        return np.clip(result, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Coordinate system
    # ------------------------------------------------------------------

    def coordinate_grid(
        self,
        height: int,
        width: int,
    ) -> tuple[np.ndarray, np.ndarray]:

        y, x = np.mgrid[0:height, 0:width]

        return (
            x.astype(np.float32),
            y.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Soft bounding box
    # ------------------------------------------------------------------

    def bounding_box_field(
        self,
        width: int,
        height: int,
        box,
    ) -> np.ndarray:

        if box is None:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        x, y = self.coordinate_grid(height, width)

        x0 = float(np.clip(box.x0, 0, width - 1))
        y0 = float(np.clip(box.y0, 0, height - 1))
        x1 = float(np.clip(box.x1, 0, width - 1))
        y1 = float(np.clip(box.y1, 0, height - 1))

        if x1 < x0:
            x0, x1 = x1, x0

        if y1 < y0:
            y0, y1 = y1, y0

        dx = np.maximum(
            np.maximum(x0 - x, 0.0),
            x - x1,
        )

        dy = np.maximum(
            np.maximum(y0 - y, 0.0),
            y - y1,
        )

        distance = np.sqrt(
            dx * dx +
            dy * dy
        )

        # Resolution-relative transition width.
        scale = max(
            1.0,
            min(width, height) * 0.015,
        )

        field = np.exp(
            -(distance * distance)
            / (2.0 * scale * scale)
        )

        confidence = float(
            np.clip(
                getattr(box, "confidence", 1.0),
                0.0,
                1.0,
            )
        )

        return np.clip(
            field * confidence,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Landmark Gaussian
    # ------------------------------------------------------------------

    def landmark_gaussian(
        self,
        width: int,
        height: int,
        point,
        sigma: float,
    ) -> np.ndarray:

        if point is None:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        confidence = float(
            np.clip(
                getattr(point, "confidence", 1.0),
                0.0,
                1.0,
            )
        )

        if confidence <= 0.0:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        x, y = self.coordinate_grid(height, width)

        px = float(point.x)
        py = float(point.y)

        sigma = max(float(sigma), 1e-3)

        distance_sq = (
            (x - px) ** 2 +
            (y - py) ** 2
        )

        field = np.exp(
            -distance_sq /
            (2.0 * sigma * sigma)
        )

        return np.clip(
            field * confidence,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Landmark access
    # ------------------------------------------------------------------

    @staticmethod
    def get_landmark(
        landmarks,
        index: int,
    ):
        if landmarks is None:
            return None

        if index < 0 or index >= len(landmarks):
            return None

        return landmarks[index]

    # ------------------------------------------------------------------
    # Feature field construction
    # ------------------------------------------------------------------

    def feature_landmark_field(
        self,
        width: int,
        height: int,
        landmarks,
        indices: tuple[int, ...],
        sigma: float,
    ) -> np.ndarray:

        fields = []

        for index in indices:
            point = self.get_landmark(
                landmarks,
                index,
            )

            if point is None:
                continue

            fields.append(
                self.landmark_gaussian(
                    width,
                    height,
                    point,
                    sigma,
                )
            )

        if not fields:
            return np.zeros(
                (height, width),
                dtype=np.float32,
            )

        return np.maximum.reduce(fields).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def smooth_field(
        self,
        field: np.ndarray,
        sigma: float = 1.5,
    ) -> np.ndarray:

        sigma = max(float(sigma), 0.0)

        if sigma <= 0.0:
            return np.clip(
                field,
                0.0,
                1.0,
            ).astype(np.float32)

        result = cv2.GaussianBlur(
            field.astype(np.float32),
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Main fields
    # ------------------------------------------------------------------

    def calculate_face_field(
        self,
        width: int,
        height: int,
        observation: GeometryObservation,
    ) -> np.ndarray:

        field = self.bounding_box_field(
            width,
            height,
            observation.face_box,
        )

        if observation.person_mask is not None:
            mask = observation.person_mask.astype(
                np.float32
            )

            if mask.shape != (height, width):
                mask = cv2.resize(
                    mask,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )

            if mask.max(initial=0.0) > 1.0:
                mask /= 255.0

            # Person mask acts as a conservative upper bound.
            field *= np.clip(
                mask,
                0.0,
                1.0,
            )

        return np.clip(
            field,
            0.0,
            1.0,
        ).astype(np.float32)

    def calculate_eye_field(
        self,
        width: int,
        height: int,
        observation: GeometryObservation,
    ) -> np.ndarray:

        return self.feature_landmark_field(
            width,
            height,
            observation.face_landmarks,
            (
                self.LEFT_EYE,
                self.RIGHT_EYE,
            ),
            sigma=max(
                2.0,
                float(
                    getattr(
                        self.style,
                        "face_landmark_sigma",
                        18.0,
                    )
                ) * 0.35,
            ),
        )

    def calculate_nose_field(
        self,
        width: int,
        height: int,
        observation: GeometryObservation,
    ) -> np.ndarray:

        return self.feature_landmark_field(
            width,
            height,
            observation.face_landmarks,
            (
                self.NOSE,
                self.NOSE_BRIDGE,
            ),
            sigma=max(
                2.0,
                float(
                    getattr(
                        self.style,
                        "face_landmark_sigma",
                        18.0,
                    )
                ) * 0.30,
            ),
        )

    def calculate_mouth_field(
        self,
        width: int,
        height: int,
        observation: GeometryObservation,
    ) -> np.ndarray:

        return self.feature_landmark_field(
            width,
            height,
            observation.face_landmarks,
            (
                self.MOUTH_LEFT,
                self.MOUTH_RIGHT,
                self.MOUTH_UPPER,
                self.MOUTH_LOWER,
            ),
            sigma=max(
                2.0,
                float(
                    getattr(
                        self.style,
                        "face_landmark_sigma",
                        18.0,
                    )
                ) * 0.30,
            ),
        )

    # ------------------------------------------------------------------
    # Feature combination
    # ------------------------------------------------------------------

    def calculate_central_feature_field(
        self,
        eye_field: np.ndarray,
        nose_field: np.ndarray,
        mouth_field: np.ndarray,
    ) -> np.ndarray:

        return np.maximum.reduce(
            [
                eye_field,
                nose_field,
                mouth_field,
            ]
        ).astype(np.float32)

    def calculate_facial_geometry_field(
        self,
        face_field: np.ndarray,
        eye_field: np.ndarray,
        nose_field: np.ndarray,
        mouth_field: np.ndarray,
    ) -> np.ndarray:

        result = (
            0.45 * face_field +
            0.25 * eye_field +
            0.10 * nose_field +
            0.20 * mouth_field
        )

        return self.smooth_field(
            np.clip(result, 0.0, 1.0),
            sigma=float(
                getattr(
                    self.style,
                    "geometry_field_smoothing",
                    2.0,
                )
            ),
        )

    # ------------------------------------------------------------------
    # Facial importance
    # ------------------------------------------------------------------

    def calculate_face_importance(
        self,
        face_field: np.ndarray,
        eye_field: np.ndarray,
        nose_field: np.ndarray,
        mouth_field: np.ndarray,
    ) -> np.ndarray:

        result = (
            0.40 * face_field +
            0.25 * eye_field +
            0.15 * nose_field +
            0.20 * mouth_field
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Detail preservation
    # ------------------------------------------------------------------

    def calculate_detail_preservation(
        self,
        face_importance: np.ndarray,
    ) -> np.ndarray:

        base = float(
            np.clip(
                getattr(
                    self.style,
                    "character_detail_retention",
                    0.34,
                ),
                0.0,
                1.0,
            )
        )

        result = (
            base +
            (1.0 - base) *
            face_importance
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def calculate_smoothing_field(
        self,
        face_field: np.ndarray,
        central_feature_field: np.ndarray,
    ) -> np.ndarray:

        strength = float(
            np.clip(
                getattr(
                    self.style,
                    "face_smoothing",
                    0.70,
                ),
                0.0,
                1.0,
            )
        )

        # Stronger on skin / face area.
        # Reduced around important facial features.
        result = (
            strength *
            face_field *
            (1.0 - 0.70 * central_feature_field)
        )

        return np.clip(
            result,
            0.0,
            1.0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Feature emphasis
    # ------------------------------------------------------------------

    def calculate_feature_emphasis(
        self,
        field: np.ndarray,
        strength: float,
    ) -> np.ndarray:

        strength = max(
            0.0,
            float(strength),
        )

        result = np.clip(
            field * strength,
            0.0,
            1.0,
        )

        return result.astype(np.float32)

    # ------------------------------------------------------------------
    # Conservative RGB application
    # ------------------------------------------------------------------

    def apply_face_control(
        self,
        frame_rgb: np.ndarray,
        detail_preservation: np.ndarray,
        smoothing_field: np.ndarray,
    ) -> np.ndarray:

        rgb = self._normalize(frame_rgb)

        # Preserve facial structure.
        detail_factor = (
            0.94 +
            0.06 * detail_preservation
        )

        result = (
            rgb *
            detail_factor[..., None]
        )

        # Very conservative smoothing approximation.
        # The main smoothing operation belongs to the
        # mathematical renderer and can later be replaced
        # by a structure-aware bilateral/domain filter.
        sigma = float(
            getattr(
                self.style,
                "smooth_sigma",
                1.15,
            )
        )

        if sigma > 0.0:
            blurred = cv2.GaussianBlur(
                result.astype(np.float32),
                ksize=(0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )

            amount = (
                0.10 *
                smoothing_field
            )[..., None]

            result = (
                result * (1.0 - amount) +
                blurred * amount
            )

        return np.clip(
            result * 255.0,
            0.0,
            255.0,
        ).round().astype(np.uint8)

    # ------------------------------------------------------------------
    # Complete transform
    # ------------------------------------------------------------------

    def transform(
        self,
        frame_rgb: np.ndarray,
        observation: GeometryObservation,
    ) -> FaceFieldResult:

        self._validate_frame(frame_rgb)
        self._validate_observation(
            observation,
            frame_rgb,
        )

        height, width = frame_rgb.shape[:2]

        face_field = self.calculate_face_field(
            width,
            height,
            observation,
        )

        eye_field = self.calculate_eye_field(
            width,
            height,
            observation,
        )

        nose_field = self.calculate_nose_field(
            width,
            height,
            observation,
        )

        mouth_field = self.calculate_mouth_field(
            width,
            height,
            observation,
        )

        central_feature_field = (
            self.calculate_central_feature_field(
                eye_field,
                nose_field,
                mouth_field,
            )
        )

        facial_geometry_field = (
            self.calculate_facial_geometry_field(
                face_field,
                eye_field,
                nose_field,
                mouth_field,
            )
        )

        face_importance = (
            self.calculate_face_importance(
                face_field,
                eye_field,
                nose_field,
                mouth_field,
            )
        )

        detail_preservation = (
            self.calculate_detail_preservation(
                face_importance,
            )
        )

        smoothing_field = (
            self.calculate_smoothing_field(
                face_field,
                central_feature_field,
            )
        )

        eye_emphasis = (
            self.calculate_feature_emphasis(
                eye_field,
                getattr(
                    self.style,
                    "eye_emphasis",
                    1.12,
                ),
            )
        )

        mouth_emphasis = (
            self.calculate_feature_emphasis(
                mouth_field,
                getattr(
                    self.style,
                    "face_contrast",
                    1.08,
                ),
            )
        )

        nose_emphasis = (
            self.calculate_feature_emphasis(
                nose_field,
                getattr(
                    self.style,
                    "face_contrast",
                    1.08,
                ),
            )
        )

        output_rgb = self.apply_face_control(
            frame_rgb,
            detail_preservation,
            smoothing_field,
        )

        return FaceFieldResult(
            output_rgb=output_rgb,
            input_rgb=frame_rgb.copy(),

            face_field=face_field,
            eye_field=eye_field,
            nose_field=nose_field,
            mouth_field=mouth_field,

            central_feature_field=central_feature_field,
            facial_geometry_field=facial_geometry_field,

            face_importance=face_importance,
            detail_preservation=detail_preservation,
            smoothing_field=smoothing_field,

            eye_emphasis=eye_emphasis,
            mouth_emphasis=mouth_emphasis,
            nose_emphasis=nose_emphasis,
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


# ======================================================================
# Compatibility Functions for Compositor and Existing Pipeline
# ======================================================================

def compute_face_mask(
    height: int,
    width: int,
    face_data: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes soft Gaussian spatial masks for face, hair, and eye regions:
        F(x,y) = exp(-((x - xc)^2 / rx^2 + (y - yc)^2 / ry^2))
    Returns:
        (face_mask, hair_mask, eye_mask): Each of shape (H, W, 1) in [0.0, 1.0] float32.
    """
    face_mask = np.zeros((height, width, 1), dtype=np.float32)
    hair_mask = np.zeros((height, width, 1), dtype=np.float32)
    eye_mask = np.zeros((height, width, 1), dtype=np.float32)

    if face_data is None:
        return face_mask, hair_mask, eye_mask

    # Extract bounding box
    bx, by, bw, bh = 0, 0, 0, 0
    if hasattr(face_data, "bbox") and face_data.bbox:
        bx = int(face_data.bbox.x * width)
        by = int(face_data.bbox.y * height)
        bw = int(face_data.bbox.width * width)
        bh = int(face_data.bbox.height * height)
    elif isinstance(face_data, dict) and "bbox" in face_data:
        b = face_data["bbox"]
        bx = int(b.get("x", 0.0) * width)
        by = int(b.get("y", 0.0) * height)
        bw = int(b.get("width", 0.0) * width)
        bh = int(b.get("height", 0.0) * height)

    if bw <= 0 or bh <= 0:
        return face_mask, hair_mask, eye_mask

    # Face center and radii
    xc = bx + bw / 2.0
    yc = by + bh / 2.0
    rx = bw * 0.48
    ry = bh * 0.52

    # Vectorized Gaussian elliptical mask
    y_coords, x_coords = np.ogrid[:height, :width]
    dist_sq = ((x_coords - xc) / max(rx, 1e-3)) ** 2 + ((y_coords - yc) / max(ry, 1e-3)) ** 2
    f_mask = np.exp(-dist_sq * 1.5).astype(np.float32)
    face_mask[:, :, 0] = np.clip(f_mask, 0.0, 1.0)

    # Hair region: upper region above face center extending upwards and laterally
    hair_yc = by + bh * 0.15
    hair_rx = bw * 0.60
    hair_ry = bh * 0.45
    h_dist = ((x_coords - xc) / max(hair_rx, 1e-3)) ** 2 + ((y_coords - hair_yc) / max(hair_ry, 1e-3)) ** 2
    h_mask = np.exp(-h_dist * 1.8).astype(np.float32)
    upper_weight = np.clip(1.0 - (y_coords - by) / max(bh * 0.65, 1e-3), 0.0, 1.0)
    hair_mask[:, :, 0] = np.clip(h_mask * upper_weight, 0.0, 1.0)

    # Eye region from landmarks or estimated bounding box
    eye_pts: List[Tuple[int, int]] = []
    if hasattr(face_data, "landmarks") and face_data.landmarks:
        lms = face_data.landmarks
        if len(lms) >= 468:
            for idx in [33, 133, 159, 145, 263, 362, 386, 374]:
                if idx < len(lms):
                    eye_pts.append((int(lms[idx].x * width), int(lms[idx].y * height)))
        elif len(lms) >= 6:
            for lm in lms[:6]:
                eye_pts.append((int(lm.x * width), int(lm.y * height)))

    if eye_pts:
        e_canvas = np.zeros((height, width), dtype=np.float32)
        eye_rad = max(4, int(bw * 0.08))
        for px, py in eye_pts:
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(e_canvas, (px, py), eye_rad, 1.0, -1)
        e_canvas = cv2.GaussianBlur(e_canvas, (0, 0), sigmaX=eye_rad * 0.7)
        eye_mask[:, :, 0] = np.clip(e_canvas, 0.0, 1.0)
    else:
        eye_yc = by + bh * 0.38
        eye_lx = bx + bw * 0.32
        eye_rx = bx + bw * 0.68
        eye_r = max(4, bw * 0.09)
        e_dist_l = ((x_coords - eye_lx) ** 2 + (y_coords - eye_yc) ** 2) / (eye_r ** 2)
        e_dist_r = ((x_coords - eye_rx) ** 2 + (y_coords - eye_yc) ** 2) / (eye_r ** 2)
        e_mask = np.exp(-e_dist_l * 1.5) + np.exp(-e_dist_r * 1.5)
        eye_mask[:, :, 0] = np.clip(e_mask.astype(np.float32), 0.0, 1.0)

    return face_mask, hair_mask, eye_mask


def apply_face_modulation(
    current_art: np.ndarray,
    original_rgb_f: np.ndarray,
    face_mask: np.ndarray,
    eye_mask: np.ndarray,
    hair_mask: np.ndarray,
    style: MathematicalAnimeStyle,
) -> np.ndarray:
    """
    Applies anime facial simplification, skin smoothing, and eye structural enhancement.
    """
    if np.max(face_mask) <= 1e-4:
        return current_art

    h, w, c = current_art.shape
    result = current_art.copy()

    # 1. Face Skin Smoothing
    skin_smooth_val = getattr(style, "skin_smoothing", 0.0)
    if skin_smooth_val > 0.0:
        face_smooth = cv2.bilateralFilter(
            (current_art * 255.0).astype(np.uint8),
            d=9,
            sigmaColor=55,
            sigmaSpace=55,
        ).astype(np.float32) / 255.0

        face_c = np.clip((face_smooth - 0.5) * style.face_contrast + 0.5, 0.0, 1.0)
        effective_face_weight = face_mask * skin_smooth_val
        result = (1.0 - effective_face_weight) * result + effective_face_weight * face_c

    # 2. Eye Structural Emphasis
    if style.eye_emphasis > 1.0 and np.max(eye_mask) > 1e-4:
        eye_weight = eye_mask * (style.eye_emphasis - 1.0)
        lum = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        is_dark = (lum < 0.35)[:, :, np.newaxis].astype(np.float32)
        is_light = (lum > 0.65)[:, :, np.newaxis].astype(np.float32)

        result = result * (1.0 - eye_weight * is_dark * 0.35)
        result = np.clip(result + eye_weight * is_light * 0.15, 0.0, 1.0)

    # 3. Hair Treatment
    if np.max(hair_mask) > 1e-4:
        hair_weight = hair_mask * 0.15
        result = np.clip(result * (1.0 - hair_weight), 0.0, 1.0)

    return np.clip(result, 0.0, 1.0)
