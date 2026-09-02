from __future__ import annotations

import cv2
import numpy as np


def normalize_points(points, width: int, height: int):
    result = []

    for point in points:
        if isinstance(point, dict):
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
        else:
            x = float(getattr(point, "x", 0.0))
            y = float(getattr(point, "y", 0.0))

        px = int(np.clip(x, 0.0, 1.0) * (width - 1))
        py = int(np.clip(y, 0.0, 1.0) * (height - 1))

        result.append((px, py))

    return result


def draw_pose_map(
    frame_shape,
    pose_points,
    connections=None,
):
    height, width = frame_shape[:2]

    canvas = np.zeros((height, width), dtype=np.uint8)

    points = normalize_points(pose_points, width, height)

    for x, y in points:
        cv2.circle(canvas, (x, y), 4, 255, -1)

    if connections:
        for a, b in connections:
            if a >= len(points) or b >= len(points):
                continue

            cv2.line(
                canvas,
                points[a],
                points[b],
                255,
                2,
                cv2.LINE_AA,
            )

    return canvas


def build_edge_map(frame, low=80, high=160):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return cv2.Canny(
        blurred,
        low,
        high,
    )


def combine_control_maps(
    edge_map,
    pose_map=None,
    face_map=None,
):
    result = edge_map.copy()

    if pose_map is not None:
        result = cv2.max(result, pose_map)

    if face_map is not None:
        result = cv2.max(result, face_map)

    return result
