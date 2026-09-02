from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from .control_maps import (
    build_edge_map,
    combine_control_maps,
    draw_pose_map,
)
from .types import ControlMap, StyleConfig


class StyleController:
    def __init__(self, config: StyleConfig | None = None):
        self.config = config or StyleConfig()

    def build_control_map(
        self,
        frame: np.ndarray,
        vision_frame: dict | Any,
    ) -> ControlMap:
        edge_map = build_edge_map(
            frame,
            low=self.config.canny_low,
            high=self.config.canny_high,
        )

        pose_map = None

        if isinstance(vision_frame, dict):
            pose = vision_frame.get("pose")
        else:
            pose = getattr(vision_frame, "pose", None)

        if pose:
            if isinstance(pose, dict):
                points = pose.get("landmarks", [])
            else:
                points = getattr(pose, "landmarks", [])

            if points:
                pose_map = draw_pose_map(
                    frame.shape,
                    points,
                )

        face_map = None

        if isinstance(vision_frame, dict):
            face = vision_frame.get("face")
            if not face and vision_frame.get("faces"):
                faces_list = vision_frame.get("faces")
                if len(faces_list) > 0:
                    face = faces_list[0]
        else:
            face = getattr(vision_frame, "face", None)
            if not face and hasattr(vision_frame, "faces") and vision_frame.faces:
                face = vision_frame.faces[0]

        if face:
            if isinstance(face, dict):
                bbox = face.get("bbox")
            else:
                bbox = getattr(face, "bbox", None)

            if bbox:
                face_map = self._build_face_map(
                    frame.shape,
                    bbox,
                )

        combined = combine_control_maps(
            edge_map=edge_map,
            pose_map=pose_map,
            face_map=face_map,
        )

        return ControlMap(
            combined_control=combined,
            edge_map=edge_map,
            pose_map=pose_map,
            face_map=face_map,
        )

    @staticmethod
    def _build_face_map(
        frame_shape,
        bbox,
    ):
        height, width = frame_shape[:2]

        canvas = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        if isinstance(bbox, dict):
            x = int(bbox.get("x", 0) * width)
            y = int(bbox.get("y", 0) * height)
            w = int(bbox.get("width", 0) * width)
            h = int(bbox.get("height", 0) * height)
        else:
            x = int(getattr(bbox, "x", 0) * width)
            y = int(getattr(bbox, "y", 0) * height)
            w = int(getattr(bbox, "width", 0) * width)
            h = int(getattr(bbox, "height", 0) * height)

        cv2.rectangle(
            canvas,
            (x, y),
            (x + w, y + h),
            255,
            2,
        )

        return canvas
