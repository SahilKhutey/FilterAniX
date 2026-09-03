from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

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


def calculate_flow(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
) -> np.ndarray:
    previous_gray = cv2.cvtColor(
        previous_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    return cv2.calcOpticalFlowFarneback(
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


def calculate_scene_cut(
    previous_rgb: np.ndarray,
    current_rgb: np.ndarray,
) -> bool:
    previous_gray = cv2.cvtColor(
        previous_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    current_gray = cv2.cvtColor(
        current_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    previous_mean = float(
        np.mean(previous_gray)
    )

    current_mean = float(
        np.mean(current_gray)
    )

    mean_difference = abs(
        current_mean -
        previous_mean
    ) / 255.0

    # Conservative initial threshold.
    return mean_difference > 0.35


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run MTH-10 Mathematical Temporal "
            "Field over a video."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{output_path}"
        )

    style = MathematicalAnimeStyle.creator_anime()

    engine = MathematicalTemporalField(
        style
    )

    previous_rgb = None

    processed = 0

    start_time = time.perf_counter()

    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            ok, frame_bgr = capture.read()

            if not ok:
                break

            current_rgb = cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )

            if previous_rgb is None:
                result = engine.transform(
                    current_rgb,
                    TemporalObservation(),
                )

            else:
                flow = calculate_flow(
                    previous_rgb,
                    current_rgb,
                )

                scene_cut = calculate_scene_cut(
                    previous_rgb,
                    current_rgb,
                )

                result = engine.transform(
                    current_rgb,
                    TemporalObservation(
                        optical_flow=flow,
                        scene_cut=scene_cut,
                    ),
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

            writer.write(
                output_bgr
            )

            previous_rgb = current_rgb.copy()

            processed += 1

            if processed % 10 == 0:
                elapsed = (
                    time.perf_counter() -
                    start_time
                )

                rate = (
                    processed / elapsed
                    if elapsed > 0
                    else 0.0
                )

                if frame_count > 0:
                    percent = (
                        processed /
                        frame_count *
                        100.0
                    )

                    print(
                        f"\r"
                        f"{processed}/{frame_count} "
                        f"({percent:5.1f}%) "
                        f"{rate:5.2f} FPS",
                        end="",
                    )
                else:
                    print(
                        f"\r"
                        f"{processed} frames "
                        f"{rate:5.2f} FPS",
                        end="",
                    )

    finally:
        capture.release()
        writer.release()

    elapsed = (
        time.perf_counter() -
        start_time
    )

    print()
    print()
    print("MTH-10 video processing complete.")
    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    print(f"Frames:    {processed}")
    print(f"Resolution:{width}x{height}")
    print(f"FPS:       {fps:.3f}")
    print(
        f"Runtime:   {elapsed:.2f} seconds"
    )

    if elapsed > 0:
        print(
            f"Speed:     "
            f"{processed / elapsed:.2f} FPS"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
