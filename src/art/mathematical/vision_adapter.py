from __future__ import annotations

from typing import Any, Optional
import numpy as np

from .geometry_types import (
    GeometryBox,
    GeometryObservation,
    GeometryPoint,
)


def _to_pixel_coord(val: float, dimension: int) -> float:
    """Converts normalized coordinate to pixel coordinate if needed."""
    if 0.0 <= val <= 1.0:
        return float(val * dimension)
    return float(val)


def adapt_vision_frame(
    vision_frame: Any,
    person_mask: Optional[np.ndarray] = None,
) -> GeometryObservation:
    """
    Adapts a Phase-2 VisionFrame (or compatible object/dict) into
    a clean GeometryObservation for MTH-07.
    """
    width = int(getattr(vision_frame, "width", 0))
    height = int(getattr(vision_frame, "height", 0))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid dimensions in vision frame: {width}x{height}"
        )

    # 1. Face box
    face_box: Optional[GeometryBox] = None
    face_obs = getattr(vision_frame, "face", None)
    if face_obs is not None and getattr(face_obs, "detected", False):
        bbox = getattr(face_obs, "bbox", None)
        if bbox is not None:
            bx = _to_pixel_coord(bbox.x, width)
            by = _to_pixel_coord(bbox.y, height)
            bw = _to_pixel_coord(bbox.width, width)
            bh = _to_pixel_coord(bbox.height, height)
            conf = float(
                getattr(face_obs, "confidence", 1.0)
                or getattr(bbox, "confidence", 1.0)
                or 1.0
            )
            face_box = GeometryBox(
                x0=bx,
                y0=by,
                x1=bx + bw,
                y1=by + bh,
                confidence=conf,
            )

    # 2. Face landmarks
    face_landmarks: list[GeometryPoint] = []
    if face_obs is not None:
        raw_pts = getattr(face_obs, "landmarks", []) or []
        for pt in raw_pts:
            px = _to_pixel_coord(pt.x, width)
            py = _to_pixel_coord(pt.y, height)
            p_conf = float(getattr(pt, "visibility", 1.0) or 1.0)
            face_landmarks.append(GeometryPoint(x=px, y=py, confidence=p_conf))

    # 3. Pose landmarks
    pose_landmarks: list[GeometryPoint] = []
    pose_obs = getattr(vision_frame, "pose", None)
    if pose_obs is not None and getattr(pose_obs, "detected", False):
        raw_pts = getattr(pose_obs, "landmarks", []) or []
        for pt in raw_pts:
            px = _to_pixel_coord(pt.x, width)
            py = _to_pixel_coord(pt.y, height)
            p_conf = float(getattr(pt, "visibility", 1.0) or 1.0)
            pose_landmarks.append(GeometryPoint(x=px, y=py, confidence=p_conf))

    # 4. Hand landmarks
    hand_landmarks: list[GeometryPoint] = []
    hands_list = getattr(vision_frame, "hands", []) or []
    for hand_obs in hands_list:
        raw_pts = getattr(hand_obs, "landmarks", []) or []
        for pt in raw_pts:
            px = _to_pixel_coord(pt.x, width)
            py = _to_pixel_coord(pt.y, height)
            p_conf = float(getattr(pt, "visibility", 1.0) or 1.0)
            hand_landmarks.append(GeometryPoint(x=px, y=py, confidence=p_conf))

    return GeometryObservation(
        width=width,
        height=height,
        face_box=face_box,
        face_landmarks=face_landmarks if face_landmarks else None,
        pose_landmarks=pose_landmarks if pose_landmarks else None,
        hand_landmarks=hand_landmarks if hand_landmarks else None,
        person_mask=person_mask,
    )
