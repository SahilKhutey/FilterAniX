"""Stage 10: Differential Texture Suppression and Background Simplification."""
from __future__ import annotations

from typing import Any, Optional
import cv2
import numpy as np

from .config import MathematicalAnimeStyle


def compute_foreground_mask(
    height: int,
    width: int,
    vision_data: Optional[Any],
) -> np.ndarray:
    """
    Constructs soft foreground mask (person / character vs background).
    Returns (H, W, 1) float32 in [0.0, 1.0].
    """
    fg_mask = np.zeros((height, width, 1), dtype=np.float32)

    if vision_data is None:
        # Default center weighted prior if vision data is absent
        y_coords, x_coords = np.ogrid[:height, :width]
        dist = ((x_coords - width / 2.0) / (width * 0.45)) ** 2 + ((y_coords - height * 0.6) / (height * 0.55)) ** 2
        prior = np.exp(-dist * 1.2).astype(np.float32)
        fg_mask[:, :, 0] = np.clip(prior, 0.0, 1.0)
        return fg_mask

    # Check for segmentation person_mask
    mask_obj = getattr(vision_data, "person_mask", None)
    if mask_obj is not None:
        if hasattr(mask_obj, "mask") and mask_obj.mask is not None:
            raw_m = mask_obj.mask
            if raw_m.shape[:2] != (height, width):
                raw_m = cv2.resize(raw_m, (width, height), interpolation=cv2.INTER_LINEAR)
            fg_mask[:, :, 0] = np.clip(raw_m.astype(np.float32) / (255.0 if raw_m.max() > 1.0 else 1.0), 0.0, 1.0)
            return fg_mask

    # Fallback to face / pose bounding boxes
    boxes = []
    faces = getattr(vision_data, "faces", []) or []
    for f in faces:
        b = getattr(f, "bbox", None) or (f.get("bbox") if isinstance(f, dict) else None)
        if b:
            boxes.append(b)

    pose = getattr(vision_data, "pose", None) or (vision_data.get("pose") if isinstance(vision_data, dict) else None)
    if pose:
        b = getattr(pose, "bbox", None) or (pose.get("bbox") if isinstance(pose, dict) else None)
        if b:
            boxes.append(b)

    if boxes:
        canvas = np.zeros((height, width), dtype=np.float32)
        for b in boxes:
            bx = int(getattr(b, "x", b.get("x", 0.0) if isinstance(b, dict) else 0.0) * width)
            by = int(getattr(b, "y", b.get("y", 0.0) if isinstance(b, dict) else 0.0) * height)
            bw = int(getattr(b, "width", b.get("width", 0.0) if isinstance(b, dict) else 0.0) * width)
            bh = int(getattr(b, "height", b.get("height", 0.0) if isinstance(b, dict) else 0.0) * height)
            # Expand slightly
            pad_w = int(bw * 0.2)
            pad_h = int(bh * 0.2)
            x1 = max(0, bx - pad_w)
            y1 = max(0, by - pad_h)
            x2 = min(width, bx + bw + pad_w)
            y2 = min(height, by + bh + pad_h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 1.0, -1)
        # Smooth boundaries
        canvas = cv2.GaussianBlur(canvas, (0, 0), sigmaX=width * 0.03)
        fg_mask[:, :, 0] = np.clip(canvas, 0.0, 1.0)
    else:
        # Generic center-bottom foreground assumption
        fg_mask[:, :, 0] = 0.5

    return fg_mask


def apply_background_simplification(
    art_field: np.ndarray,
    foreground_mask: np.ndarray,
    style: MathematicalAnimeStyle,
) -> np.ndarray:
    """
    Suppresses photographic texture in background while retaining crisp character details:
        C_{bg}' = f(C_{bg}, beta_bg) where beta_bg > beta_character
    Returns:
        Simplified field in [0.0, 1.0] RGB.
    """
    if style.background_simplification <= 0.0:
        return art_field

    # Background mask is the inverse of foreground mask
    bg_mask = np.clip(1.0 - foreground_mask, 0.0, 1.0)

    # Simplified background using strong bilateral + edge-preserving smoothing
    bg_uint8 = (art_field * 255.0).astype(np.uint8)
    simplified_bg = cv2.bilateralFilter(
        bg_uint8,
        d=11,
        sigmaColor=75,
        sigmaSpace=75,
    ).astype(np.float32) / 255.0

    # Smooth blend between character foreground and simplified background
    weight = bg_mask * style.background_simplification
    result = (1.0 - weight) * art_field + weight * simplified_bg

    return np.clip(result, 0.0, 1.0)
