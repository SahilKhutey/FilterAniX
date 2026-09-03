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
    MathematicalShadowHighlightField,
)


def save_gray(
    path: Path,
    image: np.ndarray,
) -> None:

    normalized = cv2.normalize(
        image.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    cv2.imwrite(
        str(path),
        normalized.astype(np.uint8),
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

    engine = (
        MathematicalShadowHighlightField()
    )

    result = engine.transform(
        rgb
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

    save_gray(
        Path(
            f"{stem}_illumination.png"
        ),
        result.illumination_field,
    )

    save_gray(
        Path(
            f"{stem}_shadow_probability.png"
        ),
        result.shadow_probability,
    )

    save_gray(
        Path(
            f"{stem}_highlight_probability.png"
        ),
        result.highlight_probability,
    )

    save_gray(
        Path(
            f"{stem}_shadow_field.png"
        ),
        result.shadow_field,
    )

    save_gray(
        Path(
            f"{stem}_highlight_field.png"
        ),
        result.highlight_field,
    )

    save_gray(
        Path(
            f"{stem}_target_luminance.png"
        ),
        result.target_luminance,
    )

    print()
    print(
        "MTH-06 Mathematical "
        "Shadow / Highlight Field"
    )
    print("--------------------------------")

    print(
        f"Input:       {input_path}"
    )

    print(
        f"Output:      {output_path}"
    )

    print(
        f"Resolution:  "
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        f"Mean illumination: "
        f"{result.illumination_field.mean():.6f}"
    )

    print(
        f"Mean shadow: "
        f"{result.shadow_field.mean():.6f}"
    )

    print(
        f"Mean highlight: "
        f"{result.highlight_field.mean():.6f}"
    )

    print(
        f"Mean target luminance: "
        f"{result.target_luminance.mean():.6f}"
    )

    print()
    print(
        "Diagnostic fields written."
    )


if __name__ == "__main__":
    main()
