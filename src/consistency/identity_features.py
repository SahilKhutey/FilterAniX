from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class IdentityFeatures:
    face_crop: Optional[np.ndarray]

    face_histogram: np.ndarray

    face_gray: np.ndarray

    geometry: np.ndarray

    color_histogram: np.ndarray

    edge_density: float


def _normalize_vector(
    value: np.ndarray,
) -> np.ndarray:

    value = np.asarray(
        value,
        dtype=np.float32,
    )

    norm = np.linalg.norm(value)

    if norm < 1e-8:
        return value

    return value / norm


def crop_face(
    image_rgb: np.ndarray,
    bbox: Optional[dict],
    padding: float = 0.25,
) -> Optional[np.ndarray]:

    if bbox is None:
        return None

    h, w = image_rgb.shape[:2]

    # Support multiple bbox formats: dict with x1,y1,x2,y2 or x,y,width,height
    if "x1" in bbox and "y1" in bbox:
        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox.get("x2", x1 + bbox.get("width", 0)))
        y2 = int(bbox.get("y2", y1 + bbox.get("height", 0)))
    elif "x" in bbox and "y" in bbox:
        bx = float(bbox["x"])
        by = float(bbox["y"])
        bw = float(bbox.get("width", 0))
        bh = float(bbox.get("height", 0))
        # Handle normalized vs pixel coordinates
        if bw <= 1.0 and bh <= 1.0 and w > 1:
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
        else:
            x1, y1 = int(bx), int(by)
            x2, y2 = int(bx + bw), int(by + bh)
    else:
        return None

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    px = int(bw * padding)
    py = int(bh * padding)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    if x2 <= x1 or y2 <= y1:
        return None

    return image_rgb[
        y1:y2,
        x1:x2,
    ].copy()


def extract_identity_features(
    image_rgb: np.ndarray,
    face_bbox: Optional[dict] = None,
    landmarks: Optional[list] = None,
) -> IdentityFeatures:

    face = crop_face(
        image_rgb,
        face_bbox,
    )

    if face is None:
        face = image_rgb.copy()

    face = cv2.resize(
        face,
        (128, 128),
        interpolation=cv2.INTER_AREA,
    )

    hsv = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2HSV,
    )

    face_hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    cv2.normalize(
        face_hist,
        face_hist,
    )

    face_gray = cv2.cvtColor(
        face,
        cv2.COLOR_RGB2GRAY,
    )

    face_gray = cv2.equalizeHist(
        face_gray,
    )

    face_gray = cv2.GaussianBlur(
        face_gray,
        (3, 3),
        0,
    )

    geometry = []

    if landmarks:

        for point in landmarks:

            if isinstance(point, dict):

                geometry.extend([
                    float(point.get("x", 0.0)),
                    float(point.get("y", 0.0)),
                ])

            elif hasattr(point, "x") and hasattr(point, "y"):

                geometry.extend([
                    float(point.x),
                    float(point.y),
                ])

            elif isinstance(point, (list, tuple)):

                if len(point) >= 2:
                    geometry.extend([
                        float(point[0]),
                        float(point[1]),
                    ])

    geometry = _normalize_vector(
        np.asarray(
            geometry,
            dtype=np.float32,
        )
    )

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    color_hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [32],
        [0, 256],
    )

    cv2.normalize(
        color_hist,
        color_hist,
    )

    edges = cv2.Canny(
        gray,
        50,
        150,
    )

    edge_density = float(
        np.count_nonzero(edges)
        / max(1, edges.size)
    )

    return IdentityFeatures(
        face_crop=face,
        face_histogram=face_hist,
        face_gray=face_gray,
        geometry=geometry,
        color_histogram=color_hist,
        edge_density=edge_density,
    )
