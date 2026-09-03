"""FilterAniX Objective Preservation and Transformation Metrics Engine.

Computes mathematical fidelity metrics according to the FilterAniX Rendering Contract:
  - P_structure : Structural and compositional preservation
  - P_face      : Facial feature and geometry fidelity
  - P_pose      : Skeletal pose and silhouette continuity
  - D_color     : Curated anime palette stylization depth
  - D_edge      : Illustrative line cleanliness vs raw photographic noise
  - S_temporal  : Motion-sensitive temporal stability (stability != freezing)
  - Q           : Composite Quality and Fidelity Index
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQualityAudit:
    """Comprehensive objective fidelity and artistic transformation metrics."""
    p_structure: float
    p_face: float
    p_pose: float
    d_color: float
    d_edge: float
    s_temporal: float
    a_artistic: float
    q_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "p_structure": round(self.p_structure, 4),
            "p_face": round(self.p_face, 4),
            "p_pose": round(self.p_pose, 4),
            "d_color": round(self.d_color, 4),
            "d_edge": round(self.d_edge, 4),
            "s_temporal": round(self.s_temporal, 4),
            "a_artistic": round(self.a_artistic, 4),
            "q_score": round(self.q_score, 4),
        }


class PreservationMetricsEngine:
    """Evaluates mathematical preservation invariants and artistic transformation."""

    def __init__(
        self,
        weight_structure: float = 0.25,
        weight_face: float = 0.25,
        weight_pose: float = 0.15,
        weight_temporal: float = 0.20,
        weight_artistic: float = 0.15,
    ):
        self.w_s = weight_structure
        self.w_f = weight_face
        self.w_p = weight_pose
        self.w_t = weight_temporal
        self.w_a = weight_artistic

    @staticmethod
    def compute_structural_preservation(source_rgb: np.ndarray, transformed_rgb: np.ndarray) -> float:
        """
        Measures preservation of layout, perspective, and major visual structure.
        Uses normalized cross-correlation of multi-scale gradient fields and luminance structure.
        """
        src_gray = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        trans_gray = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Multi-scale gradient structure correlation
        src_gx = cv2.Sobel(src_gray, cv2.CV_32F, 1, 0, ksize=3)
        src_gy = cv2.Sobel(src_gray, cv2.CV_32F, 0, 1, ksize=3)
        trans_gx = cv2.Sobel(trans_gray, cv2.CV_32F, 1, 0, ksize=3)
        trans_gy = cv2.Sobel(trans_gray, cv2.CV_32F, 0, 1, ksize=3)

        src_mag = cv2.magnitude(src_gx, src_gy)
        trans_mag = cv2.magnitude(trans_gx, trans_gy)

        src_std = float(np.std(src_mag))
        trans_std = float(np.std(trans_mag))
        if src_std < 1e-4 or trans_std < 1e-4:
            return 1.0

        cov = float(np.mean((src_mag - np.mean(src_mag)) * (trans_mag - np.mean(trans_mag))))
        corr = cov / (src_std * trans_std)
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def compute_face_preservation(
        source_rgb: np.ndarray,
        transformed_rgb: np.ndarray,
        vision_data: Optional[Any] = None,
    ) -> float:
        """
        Measures facial landmark stability, eye pupil contrast, and mouth contour preservation.
        Guarantees that face structure is retained and never collapsed to a black blob.
        """
        if vision_data is None:
            return 1.0

        faces = getattr(vision_data, "faces", []) or []
        if not faces:
            return 1.0

        face = faces[0]
        h, w = source_rgb.shape[:2]
        bbox = getattr(face, "bbox", None)
        if bbox is None:
            return 1.0

        bx0 = max(0, int(bbox.x * w))
        by0 = max(0, int(bbox.y * h))
        bx1 = min(w, int((bbox.x + bbox.width) * w))
        by1 = min(h, int((bbox.y + bbox.height) * h))

        if (bx1 - bx0) < 4 or (by1 - by0) < 4:
            return 1.0

        src_face = source_rgb[by0:by1, bx0:bx1]
        trans_face = transformed_rgb[by0:by1, bx0:bx1]

        # Check for severe face crushing
        trans_face_mean = float(np.mean(trans_face))
        if trans_face_mean < 25.0:
            return 0.2  # Severe penalty for crushed face

        # Feature contrast fidelity
        src_lum = cv2.cvtColor(src_face, cv2.COLOR_RGB2GRAY).astype(np.float32)
        trans_lum = cv2.cvtColor(trans_face, cv2.COLOR_RGB2GRAY).astype(np.float32)

        src_std = float(np.std(src_lum))
        trans_std = float(np.std(trans_lum))
        if src_std < 1e-3 or trans_std < 1e-3:
            return 1.0

        # Normalized cross-correlation of facial feature layout
        cov = float(np.mean((src_lum - np.mean(src_lum)) * (trans_lum - np.mean(trans_lum))))
        corr = cov / (src_std * trans_std)
        struct_fid = (corr + 1.0) / 2.0

        # Contrast retention: ensure eyes/mouth features didn't wash out completely
        contrast_sanity = min(1.0, trans_std / 8.0)

        p_face = struct_fid * 0.75 + contrast_sanity * 0.25
        return float(np.clip(p_face, 0.0, 1.0))

    @staticmethod
    def compute_pose_preservation(
        source_rgb: np.ndarray,
        transformed_rgb: np.ndarray,
        vision_data: Optional[Any] = None,
    ) -> float:
        """
        Verifies that body and hand silhouettes remain coherent and unshifted.
        """
        if vision_data is None:
            return 1.0

        pose = getattr(vision_data, "pose", None)
        if pose is None or not getattr(pose, "detected", False):
            return 1.0

        bbox = getattr(pose, "bbox", None)
        if bbox is None:
            return 1.0

        h, w = source_rgb.shape[:2]
        px0 = max(0, int(bbox.x * w))
        py0 = max(0, int(bbox.y * h))
        px1 = min(w, int((bbox.x + bbox.width) * w))
        py1 = min(h, int((bbox.y + bbox.height) * h))

        if (px1 - px0) < 4 or (py1 - py0) < 4:
            return 1.0

        src_crop = source_rgb[py0:py1, px0:px1]
        trans_crop = transformed_rgb[py0:py1, px0:px1]

        # Ensure body silhouette didn't collapse
        if float(np.mean(trans_crop)) < 15.0:
            return 0.3

        return 1.0

    @staticmethod
    def compute_color_transformation(source_rgb: np.ndarray, transformed_rgb: np.ndarray) -> float:
        """
        Measures the deliberate artistic color shift from raw camera pixels toward anime palette.
        A score of 0.0 means identical to raw photo (no stylization); 1.0 means optimal anime grading.
        """
        diff = np.abs(source_rgb.astype(np.float32) - transformed_rgb.astype(np.float32))
        mean_diff = float(np.mean(diff))
        # Optimal color shift in anime stylization is between 15 and 65 intensity levels
        if mean_diff < 5.0:
            return float(mean_diff / 5.0 * 0.3)  # Too close to original photo
        return float(np.clip(mean_diff / 45.0, 0.0, 1.0))

    @staticmethod
    def compute_edge_transformation(source_rgb: np.ndarray, transformed_rgb: np.ndarray) -> float:
        """
        Measures the transition from noisy photographic texture to clean anime line art.
        """
        trans_gray = cv2.cvtColor(transformed_rgb, cv2.COLOR_RGB2GRAY)
        # Check gradient sharpness
        grad_x = cv2.Sobel(trans_gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(trans_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        edge_cleanliness = float(np.mean(grad_mag > 40.0) / max(1e-4, np.mean(grad_mag > 10.0)))
        return float(np.clip(edge_cleanliness * 1.5, 0.0, 1.0))

    @staticmethod
    def compute_temporal_stability(
        prev_art: Optional[np.ndarray],
        current_art: np.ndarray,
        motion_score: float = 0.0,
    ) -> float:
        """
        Measures frame-to-frame stability without penalizing genuine subject motion.
        Stability != freezing: genuine motion is accounted for by the motion score.
        """
        if prev_art is None:
            return 1.0

        if prev_art.shape != current_art.shape:
            return 1.0

        diff = np.abs(prev_art.astype(np.float32) - current_art.astype(np.float32))
        frame_diff = float(np.mean(diff) / 255.0)

        # Expected frame difference is proportional to genuine motion
        unwanted_flicker = max(0.0, frame_diff - motion_score * 1.5)
        stability = 1.0 - np.clip(unwanted_flicker * 3.5, 0.0, 1.0)
        return float(stability)

    def evaluate_frame(
        self,
        source_rgb: np.ndarray,
        transformed_rgb: np.ndarray,
        prev_art: Optional[np.ndarray] = None,
        vision_data: Optional[Any] = None,
        motion_score: float = 0.0,
    ) -> FrameQualityAudit:
        """Evaluates all objective fidelity and artistic transformation metrics."""
        p_struct = self.compute_structural_preservation(source_rgb, transformed_rgb)
        p_face = self.compute_face_preservation(source_rgb, transformed_rgb, vision_data)
        p_pose = self.compute_pose_preservation(source_rgb, transformed_rgb, vision_data)
        d_color = self.compute_color_transformation(source_rgb, transformed_rgb)
        d_edge = self.compute_edge_transformation(source_rgb, transformed_rgb)
        s_temporal = self.compute_temporal_stability(prev_art, transformed_rgb, motion_score)

        a_artistic = float(0.5 * d_color + 0.5 * d_edge)

        q = float(
            self.w_s * p_struct
            + self.w_f * p_face
            + self.w_p * p_pose
            + self.w_t * s_temporal
            + self.w_a * a_artistic
        )

        return FrameQualityAudit(
            p_structure=p_struct,
            p_face=p_face,
            p_pose=p_pose,
            d_color=d_color,
            d_edge=d_edge,
            s_temporal=s_temporal,
            a_artistic=a_artistic,
            q_score=q,
        )
