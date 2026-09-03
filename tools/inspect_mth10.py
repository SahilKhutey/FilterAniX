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
    MathematicalAnimeStyle,
    MathematicalTemporalField,
    TemporalObservation,
)


def save_gray(
    path: Path,
    field: np.ndarray,
) -> None:
    image = np.clip(
        field * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)

    cv2.imwrite(
        str(path),
        image,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect MTH-10 Mathematical Temporal Field."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Current mathematical RGB frame.",
    )

    parser.add_argument(
        "--previous",
        required=False,
        help="Previous mathematical RGB frame.",
    )

    parser.add_argument(
        "--output",
        default="mth10_output.png",
    )

    parser.add_argument(
        "--flow-output",
        default="mth10_flow_magnitude.png",
    )

    parser.add_argument(
        "--confidence-output",
        default="mth10_temporal_confidence.png",
    )

    parser.add_argument(
        "--strength-output",
        default="mth10_temporal_strength.png",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    current_bgr = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if current_bgr is None:
        raise RuntimeError(
            f"Could not read {input_path}"
        )

    current_rgb = cv2.cvtColor(
        current_bgr,
        cv2.COLOR_BGR2RGB,
    )

    style = MathematicalAnimeStyle.creator_anime()

    engine = MathematicalTemporalField(
        style
    )

    if args.previous:
        previous_bgr = cv2.imread(
            args.previous,
            cv2.IMREAD_COLOR,
        )

        if previous_bgr is None:
            raise RuntimeError(
                f"Could not read {args.previous}"
            )

        previous_rgb = cv2.cvtColor(
            previous_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if previous_rgb.shape != current_rgb.shape:
            raise ValueError(
                "Current and previous images "
                "must have the same resolution."
            )

        # Calculate optical flow in grayscale.
        previous_gray = cv2.cvtColor(
            previous_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        current_gray = cv2.cvtColor(
            current_rgb,
            cv2.COLOR_RGB2GRAY,
        )

        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )

        # Establish previous frame.
        engine.transform(
            previous_rgb,
            TemporalObservation(),
        )

        result = engine.transform(
            current_rgb,
            TemporalObservation(
                optical_flow=flow,
            ),
        )

    else:
        result = engine.transform(
            current_rgb,
            TemporalObservation(),
        )

    output_rgb = np.clip(
        result.output_rgb * 255.0 + 0.5,
        0,
        255,
    ).astype(np.uint8)

    output_bgr = cv2.cvtColor(
        output_rgb,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        args.output,
        output_bgr,
    )

    save_gray(
        Path(args.flow_output),
        np.clip(
            result.flow_magnitude /
            max(style.temporal_motion_limit, 1e-6),
            0.0,
            1.0,
        ),
    )

    save_gray(
        Path(args.confidence_output),
        result.temporal_confidence,
    )

    save_gray(
        Path(args.strength_output),
        result.temporal_strength,
    )

    print()
    print("MTH-10 inspection complete.")
    print()
    print(f"Output:              {args.output}")
    print(f"Flow magnitude:      {args.flow_output}")
    print(f"Temporal confidence: {args.confidence_output}")
    print(f"Temporal strength:   {args.strength_output}")
    print()
    print(
        "Mean flow magnitude:",
        float(np.mean(result.flow_magnitude)),
    )
    print(
        "Mean confidence:",
        float(np.mean(result.temporal_confidence)),
    )
    print(
        "Mean temporal strength:",
        float(np.mean(result.temporal_strength)),
    )
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
