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
    MathematicalAnimeStyle,
    MathematicalLightingField,
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

    observation = GeometryObservation(
        width=width,
        height=height,

        face_box=GeometryBox(
            x0=width * 0.30,
            y0=height * 0.20,
            x1=width * 0.70,
            y1=height * 0.75,
            confidence=1.0,
        ),

        face_landmarks=[],
        pose_landmarks=[],
        hand_landmarks=[],

        person_mask=np.ones(
            (height, width),
            dtype=np.float32,
        ),
    )

    engine = MathematicalLightingField(
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

    field_dir = (
        output_path.parent /
        f"{output_path.stem}_mth09_fields"
    )

    field_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_field(
        field_dir / "luminance.png",
        result.luminance,
    )

    save_field(
        field_dir / "local_luminance.png",
        result.local_luminance,
    )

    # Local detail contains negative values,
    # therefore remap [-1,1] -> [0,1].
    detail = (
        result.local_detail + 1.0
    ) * 0.5

    save_field(
        field_dir / "local_detail.png",
        np.clip(detail, 0.0, 1.0),
    )

    save_field(
        field_dir / "shadow.png",
        result.shadow_field,
    )

    save_field(
        field_dir / "highlight.png",
        result.highlight_field,
    )

    save_field(
        field_dir / "midtone.png",
        result.midtone_field,
    )

    save_field(
        field_dir / "local_light.png",
        result.local_light_field,
    )

    save_field(
        field_dir / "warm_light.png",
        result.warm_light_field,
    )

    save_field(
        field_dir / "face_protection.png",
        result.face_protection_field,
    )

    save_field(
        field_dir / "highlight_protection.png",
        result.highlight_protection_field,
    )

    save_field(
        field_dir / "shadow_contribution.png",
        result.shadow_contribution,
    )

    save_field(
        field_dir / "key_light_contribution.png",
        result.key_light_contribution,
    )

    save_field(
        field_dir / "final_light_field.png",
        result.final_light_field,
    )

    print("MTH-09 completed")
    print(
        f"Input : {input_path}"
    )
    print(
        f"Output: {output_path}"
    )
    print(
        f"Resolution: {width}x{height}"
    )

    print(
        f"Luminance mean: "
        f"{result.luminance.mean():.4f}"
    )

    print(
        f"Shadow mean: "
        f"{result.shadow_field.mean():.4f}"
    )

    print(
        f"Highlight mean: "
        f"{result.highlight_field.mean():.4f}"
    )

    print(
        f"Warm light mean: "
        f"{result.warm_light_field.mean():.4f}"
    )

    print(
        f"Face protection mean: "
        f"{result.face_protection_field.mean():.4f}"
    )


if __name__ == "__main__":
    main()
