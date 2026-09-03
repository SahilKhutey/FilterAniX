from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from src.art.mathematical.palette_field import (
    MathematicalPaletteField,
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
            "FilterAniX MTH-04 "
            "Mathematical Palette Field Inspector"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="mth04_output.png",
    )

    parser.add_argument(
        "--confidence",
        default="mth04_confidence.png",
    )

    parser.add_argument(
        "--entropy",
        default="mth04_entropy.png",
    )

    parser.add_argument(
        "--dominant",
        default="mth04_dominant.png",
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

    engine = MathematicalPaletteField()

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
        args.confidence,
        result.confidence,
    )

    save_gray(
        args.entropy,
        result.palette_entropy,
    )

    palette_count = len(
        engine.style.palette
    )

    dominant = (
        result.dominant_index.astype(
            np.float32
        )
        / max(
            1,
            palette_count - 1,
        )
    )

    save_gray(
        args.dominant,
        dominant,
    )

    print("=" * 64)
    print("FilterAniX MTH-04")
    print("Mathematical Palette Field")
    print("=" * 64)

    print(
        f"Input:      {input_path}"
    )

    print(
        f"Resolution: "
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        f"Palette:    {palette_count} colors"
    )

    print(
        f"Output:     {args.output}"
    )

    print(
        f"Confidence: {args.confidence}"
    )

    print(
        f"Entropy:    {args.entropy}"
    )

    print(
        f"Dominant:   {args.dominant}"
    )

    print("-" * 64)

    print(
        "Mean palette confidence: "
        f"{result.confidence.mean():.4f}"
    )

    print(
        "Mean palette entropy: "
        f"{result.palette_entropy.mean():.4f}"
    )

    print(
        "Mean luminance before: "
        f"{result.luminance_before.mean():.4f}"
    )

    print(
        "Mean luminance after: "
        f"{result.luminance_after.mean():.4f}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()
