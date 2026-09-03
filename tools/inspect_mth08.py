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
    MathematicalAnimeStyle,
    MathematicalFaceField,
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

    landmarks = [
        GeometryPoint(
            x=width * 0.5,
            y=height * 0.5,
        )
        for _ in range(468)
    ]

    landmarks[33] = GeometryPoint(
        width * 0.40,
        height * 0.42,
    )

    landmarks[263] = GeometryPoint(
        width * 0.60,
        height * 0.42,
    )

    landmarks[1] = GeometryPoint(
        width * 0.50,
        height * 0.52,
    )

    landmarks[168] = GeometryPoint(
        width * 0.50,
        height * 0.45,
    )

    landmarks[61] = GeometryPoint(
        width * 0.43,
        height * 0.64,
    )

    landmarks[291] = GeometryPoint(
        width * 0.57,
        height * 0.64,
    )

    landmarks[13] = GeometryPoint(
        width * 0.50,
        height * 0.62,
    )

    landmarks[14] = GeometryPoint(
        width * 0.50,
        height * 0.66,
    )

    mask = np.ones(
        (height, width),
        dtype=np.float32,
    )

    return GeometryObservation(
        width=width,
        height=height,

        face_box=GeometryBox(
            x0=width * 0.30,
            y0=height * 0.25,
            x1=width * 0.70,
            y1=height * 0.78,
        ),

        face_landmarks=landmarks,

        pose_landmarks=[],
        hand_landmarks=[],

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

    input_path = Path(args.input)
    output_path = Path(args.output)

    bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise RuntimeError(
            f"Unable to read: {input_path}"
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

    engine = MathematicalFaceField(
        MathematicalAnimeStyle.creator_anime()
    )

    result = engine.transform(
        rgb,
        observation,
    )

    output_bgr = cv2.cvtColor(
        result.output_rgb,
        cv2.COLOR_RGB2BGR,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        output_bgr,
    )

    output_dir = (
        output_path.parent /
        f"{output_path.stem}_mth08_fields"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_field(
        output_dir / "face.png",
        result.face_field,
    )

    save_field(
        output_dir / "eyes.png",
        result.eye_field,
    )

    save_field(
        output_dir / "nose.png",
        result.nose_field,
    )

    save_field(
        output_dir / "mouth.png",
        result.mouth_field,
    )

    save_field(
        output_dir / "central_features.png",
        result.central_feature_field,
    )

    save_field(
        output_dir / "facial_geometry.png",
        result.facial_geometry_field,
    )

    save_field(
        output_dir / "face_importance.png",
        result.face_importance,
    )

    save_field(
        output_dir / "detail_preservation.png",
        result.detail_preservation,
    )

    save_field(
        output_dir / "smoothing.png",
        result.smoothing_field,
    )

    save_field(
        output_dir / "eye_emphasis.png",
        result.eye_emphasis,
    )

    save_field(
        output_dir / "mouth_emphasis.png",
        result.mouth_emphasis,
    )

    save_field(
        output_dir / "nose_emphasis.png",
        result.nose_emphasis,
    )

    print("MTH-08 completed")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()
    print(f"Resolution: {width}x{height}")
    print(
        f"Face mean: "
        f"{result.face_field.mean():.4f}"
    )
    print(
        f"Eye max: "
        f"{result.eye_field.max():.4f}"
    )
    print(
        f"Nose max: "
        f"{result.nose_field.max():.4f}"
    )
    print(
        f"Mouth max: "
        f"{result.mouth_field.max():.4f}"
    )
    print(
        f"Face importance mean: "
        f"{result.face_importance.mean():.4f}"
    )
    print(
        f"Detail preservation mean: "
        f"{result.detail_preservation.mean():.4f}"
    )


if __name__ == "__main__":
    main()
