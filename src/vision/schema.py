from __future__ import annotations

from typing import Any


def validate_vision_frame(data: dict[str, Any]) -> None:
    required = (
        "frame_index",
        "timestamp",
        "width",
        "height",
        "face",
        "pose",
        "hands",
        "motion",
        "scene_id",
        "scene_cut",
    )

    missing = [
        key for key in required
        if key not in data
    ]

    if missing:
        raise ValueError(
            f"Vision frame missing fields: {missing}"
        )

    if not isinstance(data["frame_index"], int):
        raise TypeError("frame_index must be int")

    if not isinstance(
        data["timestamp"],
        (int, float),
    ):
        raise TypeError(
            "timestamp must be numeric"
        )

    if data["width"] <= 0:
        raise ValueError("width must be > 0")

    if data["height"] <= 0:
        raise ValueError("height must be > 0")

    face = data["face"]

    if not isinstance(face, dict):
        raise TypeError("face must be an object")

    if not isinstance(
        face.get("detected"),
        bool,
    ):
        raise TypeError(
            "face.detected must be bool"
        )

    pose = data["pose"]

    if not isinstance(pose, dict):
        raise TypeError("pose must be an object")

    if not isinstance(
        pose.get("detected"),
        bool,
    ):
        raise TypeError(
            "pose.detected must be bool"
        )

    if not isinstance(data["hands"], list):
        raise TypeError("hands must be a list")

    motion = data["motion"]

    if not isinstance(motion, dict):
        raise TypeError(
            "motion must be an object"
        )

    if not isinstance(
        data["scene_cut"],
        bool,
    ):
        raise TypeError(
            "scene_cut must be bool"
        )
