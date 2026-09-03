from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.consistency.identity_bank import (
    IdentityReference,
    IdentityReferenceBank,
)
from src.consistency.identity_features import (
    extract_identity_features,
)
from src.consistency.identity_metric import (
    IdentityMetric,
    IdentityMetricEngine,
)
from src.consistency.reference_selector import (
    IdentityReferenceSelector,
)
from src.consistency.retry import (
    IdentityRetryPolicy,
)


@dataclass
class IdentityEvaluation:

    metric: IdentityMetric

    reference_rgb: Optional[np.ndarray]

    should_refresh_reference: bool


class IdentityEngine:

    def __init__(
        self,
        warning_threshold: float = 0.62,
        severe_threshold: float = 0.48,
        max_retries: int = 2,
        bank_size: int = 8,
    ):

        self.bank = IdentityReferenceBank(
            max_references=bank_size
        )

        self.selector = (
            IdentityReferenceSelector(
                self.bank
            )
        )

        self.metric_engine = (
            IdentityMetricEngine(
                warning_threshold=warning_threshold,
                severe_threshold=severe_threshold,
            )
        )

        self.retry_policy = (
            IdentityRetryPolicy(
                max_retries=max_retries,
                warning_threshold=warning_threshold,
                severe_threshold=severe_threshold,
            )
        )

        self.previous_features = None

    def add_initial_reference(
        self,
        frame_index: int,
        scene_id: int,
        image_rgb: np.ndarray,
        face_bbox: Optional[dict] = None,
        landmarks: Optional[list] = None,
    ) -> None:

        features = extract_identity_features(
            image_rgb,
            face_bbox,
            landmarks,
        )

        self.bank.add(
            IdentityReference(
                frame_index=frame_index,
                image=image_rgb.copy(),
                face_crop=features.face_crop,
                identity_score=1.0,
                scene_id=scene_id,
            )
        )

        self.previous_features = (
            features
        )

    def reference_for(
        self,
        scene_id: int,
        frame_index: int = 0,
    ) -> Optional[np.ndarray]:

        return self.selector.select(
            scene_id,
            frame_index,
        )

    def evaluate(
        self,
        frame_index: int,
        scene_id: int,
        image_rgb: np.ndarray,
        face_bbox: Optional[dict] = None,
        landmarks: Optional[list] = None,
    ) -> IdentityEvaluation:

        reference = self.bank.best(
            scene_id
        )

        if reference is None:

            self.add_initial_reference(
                frame_index,
                scene_id,
                image_rgb,
                face_bbox,
                landmarks,
            )

            metric = IdentityMetric(
                overall=1.0,
                face_appearance=1.0,
                face_geometry=1.0,
                color_similarity=1.0,
                edge_similarity=1.0,
                temporal_similarity=1.0,
                warning=False,
                severe_drift=False,
            )

            return IdentityEvaluation(
                metric=metric,
                reference_rgb=image_rgb.copy(),
                should_refresh_reference=False,
            )

        reference_features = (
            extract_identity_features(
                reference.image,
                face_bbox=None,
                landmarks=None,
            )
        )

        candidate_features = (
            extract_identity_features(
                image_rgb,
                face_bbox,
                landmarks,
            )
        )

        metric = self.metric_engine.compare(
            reference_features,
            candidate_features,
            self.previous_features,
        )

        self.previous_features = (
            candidate_features
        )

        should_refresh = (
            metric.overall
            >= 0.78
            and metric.overall
            > reference.identity_score
        )

        if should_refresh:

            self.selector.add_reference(
                frame_index,
                scene_id,
                image_rgb,
                metric.overall,
            )

        return IdentityEvaluation(
            metric=metric,
            reference_rgb=reference.image.copy(),
            should_refresh_reference=should_refresh,
        )
