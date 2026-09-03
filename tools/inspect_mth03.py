from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from src.art.mathematical.tone_field import (
    MathematicalToneField,
)


def save_gray(
    path: str,
    field: np.ndarray,
) -> None:

    image = (
        np.clip(
            field,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        path,
        image,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "FilterAniX MTH-03 "
            "Mathematical Tone Field Inspector"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="mth03_output.png",
    )

    parser.add_argument(
        "--luminance",
        default="mth03_luminance.png",
    )

    parser.add_argument(
        "--quantized",
        default="mth03_quantized.png",
    )

    parser.add_argument(
        "--shadow",
        default="mth03_shadow.png",
    )

    parser.add_argument(
        "--highlight",
        default="mth03_highlight.png",
    )

    parser.add_argument(
        "--target",
        default="mth03_target.png",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:

        raise RuntimeError(
            f"Unable to read {input_path}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    engine = MathematicalToneField()

    result = engine.transform(
        rgb
    )

    output_bgr = cv2.cvtColor(
        result.output_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if not cv2.imwrite(
        args.output,
        output_bgr,
    ):

        raise RuntimeError(
            f"Unable to write {args.output}"
        )

    save_gray(
        args.luminance,
        result.luminance,
    )

    save_gray(
        args.quantized,
        result.quantized_luminance,
    )

    save_gray(
        args.shadow,
        result.shadow_mask,
    )

    save_gray(
        args.highlight,
        result.highlight_mask,
    )

    save_gray(
        args.target,
        result.target_luminance,
    )

    print("=" * 64)
    print("FilterAniX MTH-03")
    print("Mathematical Luminance / Tone Field")
    print("=" * 64)

    print(
        f"Input:       {input_path}"
    )

    print(
        f"Resolution:  "
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        f"Output:      {args.output}"
    )

    print(
        f"Luminance:   {args.luminance}"
    )

    print(
        f"Quantized:   {args.quantized}"
    )

    print(
        f"Shadow:      {args.shadow}"
    )

    print(
        f"Highlight:   {args.highlight}"
    )

    print(
        f"Target:      {args.target}"
    )

    print("-" * 64)

    print(
        "Mean input luminance: "
        f"{result.luminance.mean():.4f}"
    )

    print(
        "Mean target luminance: "
        f"{result.target_luminance.mean():.4f}"
    )

    print(
        "Mean shadow field: "
        f"{result.shadow_mask.mean():.4f}"
    )

    print(
        "Mean highlight field: "
        f"{result.highlight_mask.mean():.4f}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()
