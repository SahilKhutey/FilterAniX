from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.consistency.identity_features import (
    IdentityFeatures,
)


@dataclass
class IdentityMetric:

    overall: float

    face_appearance: float

    face_geometry: float

    color_similarity: float

    edge_similarity: float

    temporal_similarity: float

    warning: bool

    severe_drift: bool


class IdentityMetricEngine:

    def __init__(
        self,
        warning_threshold: float = 0.62,
        severe_threshold: float = 0.48,
    ):

        self.warning_threshold = (
            float(warning_threshold)
        )

        self.severe_threshold = (
            float(severe_threshold)
        )

    @staticmethod
    def histogram_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        score = cv2.compareHist(
            a.astype(np.float32),
            b.astype(np.float32),
            cv2.HISTCMP_CORREL,
        )

        return float(
            np.clip(
                (score + 1.0) / 2.0,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def image_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        a = a.astype(
            np.float32
        )

        b = b.astype(
            np.float32
        )

        mse = np.mean(
            (a - b) ** 2
        )

        return float(
            np.exp(
                -mse / 1800.0
            )
        )

    @staticmethod
    def geometry_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        if a.size == 0 and b.size == 0:
            return 1.0

        if (
            a.size == 0
            or b.size == 0
            or a.shape != b.shape
        ):
            return 0.5

        distance = float(
            np.linalg.norm(a - b)
        )

        return float(
            np.exp(
                -distance
            )
        )

    def compare(
        self,
        reference: IdentityFeatures,
        candidate: IdentityFeatures,
        previous: IdentityFeatures | None = None,
    ) -> IdentityMetric:

        face_appearance = (
            0.60
            * self.histogram_similarity(
                reference.face_histogram,
                candidate.face_histogram,
            )
            +
            0.40
            * self.image_similarity(
                reference.face_gray,
                candidate.face_gray,
            )
        )

        face_geometry = (
            self.geometry_similarity(
                reference.geometry,
                candidate.geometry,
            )
        )

        color_similarity = (
            self.histogram_similarity(
                reference.color_histogram,
                candidate.color_histogram,
            )
        )

        edge_difference = abs(
            reference.edge_density
            -
            candidate.edge_density
        )

        edge_similarity = float(
            max(
                0.0,
                1.0 - edge_difference * 6.0,
            )
        )

        if previous is None:

            temporal_similarity = 1.0

        else:

            temporal_similarity = (
                self.image_similarity(
                    previous.face_gray,
                    candidate.face_gray,
                )
            )

        overall = (
            0.40 * face_appearance
            +
            0.25 * face_geometry
            +
            0.15 * color_similarity
            +
            0.10 * edge_similarity
            +
            0.10 * temporal_similarity
        )

        overall = float(
            np.clip(
                overall,
                0.0,
                1.0,
            )
        )

        return IdentityMetric(
            overall=overall,
            face_appearance=float(
                face_appearance
            ),
            face_geometry=float(
                face_geometry
            ),
            color_similarity=float(
                color_similarity
            ),
            edge_similarity=float(
                edge_similarity
            ),
            temporal_similarity=float(
                temporal_similarity
            ),
            warning=(
                overall
                < self.warning_threshold
            ),
            severe_drift=(
                overall
                < self.severe_threshold
            ),
        )
