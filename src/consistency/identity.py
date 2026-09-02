from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional
import cv2
import numpy as np

from src.consistency.types import ReferenceProfile, ConsistencyMetrics


@dataclass
class IdentityProfile:
    source: str
    width: int
    height: int
    aspect_ratio: float
    mean_color: list[float]
    hsv_histogram: list[float]
    edge_density: float


def build_identity_profile(
    image_path: str,
) -> IdentityProfile:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(image_path)

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise RuntimeError(
            f"Unable to read reference image: {image_path}"
        )

    height, width = image.shape[:2]

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    histogram = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [32, 32],
        [0, 180, 0, 256],
    )

    histogram = cv2.normalize(
        histogram,
        histogram,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edge_density = float(
        np.mean(edges > 0)
    )

    mean_color = (
        image.mean(axis=(0, 1))
        .astype(float)
        .tolist()
    )

    return IdentityProfile(
        source=str(path),
        width=width,
        height=height,
        aspect_ratio=width / max(height, 1),
        mean_color=mean_color,
        hsv_histogram=histogram.flatten().tolist(),
        edge_density=edge_density,
    )


def histogram_similarity(
    reference: IdentityProfile,
    candidate: IdentityProfile,
) -> float:
    a = np.asarray(
        reference.hsv_histogram,
        dtype=np.float32,
    )

    b = np.asarray(
        candidate.hsv_histogram,
        dtype=np.float32,
    )

    similarity = cv2.compareHist(
        a,
        b,
        cv2.HISTCMP_CORREL,
    )

    return float(
        np.clip(
            (similarity + 1.0) / 2.0,
            0.0,
            1.0,
        )
    )


class IdentityProfileBuilder:
    """Extracts lightweight visual characteristics from a canonical character reference image."""

    @staticmethod
    def build_profile(reference_rgb: np.ndarray, name: str = "creator_canonical") -> ReferenceProfile:
        h, w = reference_rgb.shape[:2]
        aspect_ratio = float(w) / float(h)

        lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean_lab = np.mean(lab, axis=(0, 1)).tolist()
        std_lab = np.std(lab, axis=(0, 1)).tolist()

        hist = cv2.calcHist([lab.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten().tolist()

        pixels = reference_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        k = 5
        try:
            _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, flags)
            dominant_palette = [[int(c[0]), int(c[1]), int(c[2])] for c in centers]
        except Exception:
            dominant_palette = [[128, 128, 128]]

        gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(h * w)

        return ReferenceProfile(
            name=name,
            color_hist=hist_flat,
            dominant_palette=dominant_palette,
            edge_density=edge_density,
            aspect_ratio=aspect_ratio,
            mean_lab=mean_lab,
            std_lab=std_lab,
        )


class IdentityScorer:
    """Evaluates the visual consistency of any generated frame against a reference profile."""

    def __init__(self, profile: ReferenceProfile, warning_threshold: float = 0.55):
        self.profile = profile
        self.warning_threshold = warning_threshold
        self.ref_hist = np.array(profile.color_hist, dtype=np.float32).reshape((8, 8, 8)) if profile.color_hist else None

    def evaluate_frame(self, frame_rgb: np.ndarray) -> ConsistencyMetrics:
        h, w = frame_rgb.shape[:2]

        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        curr_hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(curr_hist, curr_hist)

        if self.ref_hist is not None:
            hist_corr = float(cv2.compareHist(curr_hist, self.ref_hist, cv2.HISTCMP_CORREL))
            color_sim = max(0.0, min(1.0, (hist_corr + 1.0) / 2.0))
        else:
            color_sim = 0.80

        lab_f = lab.astype(np.float32)
        curr_mean = np.mean(lab_f, axis=(0, 1))
        ref_mean = np.array(self.profile.mean_lab, dtype=np.float32)
        lab_dist = float(np.linalg.norm(curr_mean - ref_mean))
        lab_sim = float(np.exp(-lab_dist / 60.0))

        combined_color_sim = 0.6 * color_sim + 0.4 * lab_sim

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        curr_edge_density = float(np.count_nonzero(edges)) / float(h * w)

        edge_diff = abs(curr_edge_density - self.profile.edge_density)
        edge_sim = max(0.0, 1.0 - min(1.0, edge_diff * 6.0))

        overall_similarity = 0.70 * combined_color_sim + 0.30 * edge_sim
        is_warning = overall_similarity < self.warning_threshold

        return ConsistencyMetrics(
            similarity=float(overall_similarity),
            color_similarity=float(combined_color_sim),
            edge_similarity=float(edge_sim),
            warning=is_warning,
        )
