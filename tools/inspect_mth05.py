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
    MathematicalEdgeField,
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

    engine = MathematicalEdgeField()

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
            f"{stem}_gradient.png"
        ),
        result.gradient_magnitude,
    )

    save_gray(
        Path(
            f"{stem}_laplacian.png"
        ),
        np.abs(
            result.laplacian
        ),
    )

    save_gray(
        Path(
            f"{stem}_multiscale.png"
        ),
        result.multiscale_response,
    )

    save_gray(
        Path(
            f"{stem}_edge_probability.png"
        ),
        result.edge_probability,
    )

    save_gray(
        Path(
            f"{stem}_line_strength.png"
        ),
        result.line_strength,
    )

    save_gray(
        Path(
            f"{stem}_line_field.png"
        ),
        result.line_field,
    )

    print()
    print("MTH-05 Mathematical Edge Field")
    print("--------------------------------")
    print(f"Input:       {input_path}")
    print(f"Output:      {output_path}")
    print(
        f"Resolution:  "
        f"{rgb.shape[1]}x{rgb.shape[0]}"
    )

    print(
        f"Mean gradient: "
        f"{result.gradient_magnitude.mean():.6f}"
    )

    print(
        f"Mean edge probability: "
        f"{result.edge_probability.mean():.6f}"
    )

    print(
        f"Mean line strength: "
        f"{result.line_strength.mean():.6f}"
    )

    print(
        f"Mean line field: "
        f"{result.line_field.mean():.6f}"
    )

    print()
    print("Diagnostic fields written.")


if __name__ == "__main__":
    main()
