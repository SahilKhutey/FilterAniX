from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from src.art.mathematical import (
    GeometryBox,
    GeometryObservation,
    GeometryPoint,
    MathematicalGeometryField,
)


def save_field(
    path: Path,
    field: np.ndarray,
) -> None:

    image = np.clip(
        field * 255.0,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(path),
        image,
    )


def synthetic_observation(
    width: int,
    height: int,
) -> GeometryObservation:

    mask = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    x0 = int(width * 0.25)
    x1 = int(width * 0.78)

    y0 = int(height * 0.10)
    y1 = int(height * 0.92)

    mask[
        y0:y1,
        x0:x1,
    ] = 1.0

    face_box = GeometryBox(
        x0=width * 0.40,
        y0=height * 0.10,
        x1=width * 0.63,
        y1=height * 0.35,
    )

    face_landmarks = [
        GeometryPoint(
            width * 0.45,
            height * 0.20,
        ),
        GeometryPoint(
            width * 0.58,
            height * 0.20,
        ),
        GeometryPoint(
            width * 0.515,
            height * 0.25,
        ),
        GeometryPoint(
            width * 0.515,
            height * 0.30,
        ),
    ]

    pose_landmarks = [
        GeometryPoint(
            width * 0.42,
            height * 0.38,
        ),
        GeometryPoint(
            width * 0.62,
            height * 0.38,
        ),
        GeometryPoint(
            width * 0.32,
            height * 0.62,
        ),
        GeometryPoint(
            width * 0.72,
            height * 0.62,
        ),
    ]

    hand_landmarks = [
        GeometryPoint(
            width * 0.25,
            height * 0.60,
        ),
        GeometryPoint(
            width * 0.78,
            height * 0.60,
        ),
    ]

    return GeometryObservation(
        width=width,
        height=height,
        face_box=face_box,
        face_landmarks=face_landmarks,
        pose_landmarks=pose_landmarks,
        hand_landmarks=hand_landmarks,
        person_mask=mask,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise FileNotFoundError(
            input_path
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    height, width = rgb.shape[:2]

    observation = synthetic_observation(
        width,
        height,
    )

    engine = MathematicalGeometryField()

    result = engine.transform(
        rgb,
        observation,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        cv2.cvtColor(
            result.output_rgb,
            cv2.COLOR_RGB2BGR,
        ),
    )

    stem = output_path.with_suffix("")

    fields = {
        "face": result.face_field,
        "face_landmark": result.face_landmark_field,
        "pose": result.pose_field,
        "hand": result.hand_field,
        "person": result.person_field,
        "character": result.character_field,
        "background": result.background_field,
        "face_importance": result.face_importance,
        "structural_importance": result.structural_importance,
        "detail_preservation": result.detail_preservation,
        "simplification": result.simplification_field,
    }

    for name, field in fields.items():

        save_field(
            Path(
                f"{stem}_{name}.png"
            ),
            field,
        )

    print()
    print(
        "MTH-07 Mathematical "
        "Character / Geometry Field"
    )
    print("--------------------------------")

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    print(
        f"Resolution: {width}x{height}"
    )

    print(
        "Mean character field:",
        f"{result.character_field.mean():.6f}",
    )

    print(
        "Mean face importance:",
        f"{result.face_importance.mean():.6f}",
    )

    print(
        "Mean detail preservation:",
        f"{result.detail_preservation.mean():.6f}",
    )

    print(
        "Mean background simplification:",
        f"{result.simplification_field.mean():.6f}",
    )

    print()
    print(
        "Diagnostic fields written."
    )


if __name__ == "__main__":
    main()
