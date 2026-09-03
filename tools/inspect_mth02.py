from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from src.art.mathematical.color_field import (
    MathematicalColorField,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Inspect FilterAniX MTH-02 "
            "Mathematical Color Field"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="mth02_output.png",
    )

    parser.add_argument(
        "--palette-output",
        default="mth02_palette.png",
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

    engine = MathematicalColorField()

    result = engine.transform(
        rgb
    )

    output_bgr = cv2.cvtColor(
        result.output_rgb,
        cv2.COLOR_RGB2BGR,
    )

    palette_bgr = cv2.cvtColor(
        result.palette_rgb,
        cv2.COLOR_RGB2BGR,
    )

    if not cv2.imwrite(
        args.output,
        output_bgr,
    ):

        raise RuntimeError(
            f"Unable to write {args.output}"
        )

    if not cv2.imwrite(
        args.palette_output,
        palette_bgr,
    ):

        raise RuntimeError(
            f"Unable to write {args.palette_output}"
        )

    print("=" * 60)
    print("FilterAniX MTH-02")
    print("Mathematical Color Field")
    print("=" * 60)

    print(
        f"Input:           {input_path}"
    )

    print(
        f"Resolution:      "
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        f"Output:          {args.output}"
    )

    print(
        f"Palette output:  "
        f"{args.palette_output}"
    )

    print(
        f"Input dtype:     {rgb.dtype}"
    )

    print(
        f"Output dtype:    "
        f"{result.output_rgb.dtype}"
    )

    print("=" * 60)
    print("MTH-02 transformation complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
