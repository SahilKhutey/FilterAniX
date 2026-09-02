"""Character Reference Profile Builder and Identity Similarity Scorer."""
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.consistency.types import ReferenceProfile, ConsistencyMetrics


class IdentityProfileBuilder:
    """Extracts lightweight visual characteristics from a canonical character reference image."""

    @staticmethod
    def build_profile(reference_rgb: np.ndarray, name: str = "creator_canonical") -> ReferenceProfile:
        h, w = reference_rgb.shape[:2]
        aspect_ratio = float(w) / float(h)

        # 1. CIELAB Mean and Standard Deviation
        lab = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        mean_lab = np.mean(lab, axis=(0, 1)).tolist()
        std_lab = np.std(lab, axis=(0, 1)).tolist()

        # 2. 3D Color Histogram (8x8x8 bins = 512 bins)
        hist = cv2.calcHist([lab.astype(np.uint8)], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten().tolist()

        # 3. Dominant Color Palette (Top 5 representative RGB centroids)
        pixels = reference_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        k = 5
        try:
            _, _, centers = cv2.kmeans(pixels, k, None, criteria, 3, flags)
            dominant_palette = [[int(c[0]), int(c[1]), int(c[2])] for c in centers]
        except Exception:
            dominant_palette = [[128, 128, 128]]

        # 4. Edge Density
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

        # 1. Color Histogram Correlation in Lab space
        lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB)
        curr_hist = cv2.calcHist([lab], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(curr_hist, curr_hist)

        if self.ref_hist is not None:
            hist_corr = float(cv2.compareHist(curr_hist, self.ref_hist, cv2.HISTCMP_CORREL))
            color_sim = max(0.0, min(1.0, (hist_corr + 1.0) / 2.0))
        else:
            color_sim = 0.80

        # 2. Lab Centroid Proximity
        lab_f = lab.astype(np.float32)
        curr_mean = np.mean(lab_f, axis=(0, 1))
        ref_mean = np.array(self.profile.mean_lab, dtype=np.float32)
        lab_dist = float(np.linalg.norm(curr_mean - ref_mean))
        lab_sim = float(np.exp(-lab_dist / 60.0))

        combined_color_sim = 0.6 * color_sim + 0.4 * lab_sim

        # 3. Edge Structure Density Similarity
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        curr_edge_density = float(np.count_nonzero(edges)) / float(h * w)
        
        edge_diff = abs(curr_edge_density - self.profile.edge_density)
        edge_sim = max(0.0, 1.0 - min(1.0, edge_diff * 6.0))

        # Overall Similarity Score
        overall_similarity = 0.70 * combined_color_sim + 0.30 * edge_sim
        is_warning = overall_similarity < self.warning_threshold

        return ConsistencyMetrics(
            similarity=float(overall_similarity),
            color_similarity=float(combined_color_sim),
            edge_similarity=float(edge_sim),
            warning=is_warning,
        )
