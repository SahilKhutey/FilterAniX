"""Stages 7, 8, 9: Face Field, Eye Structural Emphasis, and Hair Modulation."""
from __future__ import annotations

from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from .config import MathematicalAnimeStyle


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
    # Hair is predominantly in upper half
    upper_weight = np.clip(1.0 - (y_coords - by) / max(bh * 0.65, 1e-3), 0.0, 1.0)
    hair_mask[:, :, 0] = np.clip(h_mask * upper_weight, 0.0, 1.0)

    # Eye region from landmarks or estimated bounding box
    eye_pts: List[Tuple[int, int]] = []
    if hasattr(face_data, "landmarks") and face_data.landmarks:
        lms = face_data.landmarks
        if len(lms) >= 468:
            # MediaPipe eye indices: left pupil ~468, right pupil ~473, or eye centers: 33, 263, 133, 362
            for idx in [33, 133, 159, 145, 263, 362, 386, 374]:
                if idx < len(lms):
                    eye_pts.append((int(lms[idx].x * width), int(lms[idx].y * height)))
        elif len(lms) >= 6:
            for lm in lms[:6]:
                eye_pts.append((int(lm.x * width), int(lm.y * height)))

    if eye_pts:
        # Render Gaussian spots at detected eye locations
        e_canvas = np.zeros((height, width), dtype=np.float32)
        eye_rad = max(4, int(bw * 0.08))
        for px, py in eye_pts:
            if 0 <= px < width and 0 <= py < height:
                cv2.circle(e_canvas, (px, py), eye_rad, 1.0, -1)
        e_canvas = cv2.GaussianBlur(e_canvas, (0, 0), sigmaX=eye_rad * 0.7)
        eye_mask[:, :, 0] = np.clip(e_canvas, 0.0, 1.0)
    else:
        # Fallback eye estimate: upper mid quadrant of face bbox
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
    Applies anime facial simplification, skin smoothing, and eye structural enhancement:
        Face: lower texture, simplified colors, controlled contrast
        Eyes: enhanced local structural clarity and pupil contrast
        Hair: deeper cel shadows and crisp contour lines
    """
    if np.max(face_mask) <= 1e-4:
        return current_art

    h, w, c = current_art.shape
    result = current_art.copy()

    # 1. Face Skin Smoothing: Bilateral / Gaussian smoothing focused inside face mask
    if style.skin_smoothing > 0.0:
        face_smooth = cv2.bilateralFilter(
            (current_art * 255.0).astype(np.uint8),
            d=9,
            sigmaColor=55,
            sigmaSpace=55,
        ).astype(np.float32) / 255.0

        # Enhance face contrast slightly
        face_c = np.clip((face_smooth - 0.5) * style.face_contrast + 0.5, 0.0, 1.0)

        # Blend smoothed face using face mask
        effective_face_weight = face_mask * style.skin_smoothing
        result = (1.0 - effective_face_weight) * result + effective_face_weight * face_c

    # 2. Eye Structural Emphasis: Boost contrast and sharp contours in eye zones
    if style.eye_emphasis > 1.0 and np.max(eye_mask) > 1e-4:
        # Local high-pass detail in eyes from original footage
        eye_weight = eye_mask * (style.eye_emphasis - 1.0)
        # Deepen darks (pupils/eyelashes) and brighten eye highlights
        lum = 0.299 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 2]
        is_dark = (lum < 0.35)[:, :, np.newaxis].astype(np.float32)
        is_light = (lum > 0.65)[:, :, np.newaxis].astype(np.float32)

        # Deepen pupil/lash ink
        result = result * (1.0 - eye_weight * is_dark * 0.35)
        # Brighten specular reflection
        result = np.clip(result + eye_weight * is_light * 0.15, 0.0, 1.0)

    # 3. Hair Treatment: Controlled deeper shadows and highlights
    if np.max(hair_mask) > 1e-4:
        # Subtle hair shadow deepening
        hair_weight = hair_mask * 0.15
        result = np.clip(result * (1.0 - hair_weight), 0.0, 1.0)

    return np.clip(result, 0.0, 1.0)
