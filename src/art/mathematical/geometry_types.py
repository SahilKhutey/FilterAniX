from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GeometryPoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class GeometryBox:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float = 1.0


@dataclass
class GeometryObservation:
    width: int
    height: int

    face_box: GeometryBox | None = None
    face_landmarks: list[GeometryPoint] | None = None

    pose_landmarks: list[GeometryPoint] | None = None
    hand_landmarks: list[GeometryPoint] | None = None

    person_mask: np.ndarray | None = None
