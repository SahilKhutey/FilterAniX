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

from src.art.mathematical.tone_field import (
    MathematicalToneField,
)


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="mth03_video.mp4",
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    if not input_path.exists():

        raise FileNotFoundError(
            input_path
        )

    capture = cv2.VideoCapture(
        str(input_path)
    )

    if not capture.isOpened():

        raise RuntimeError(
            f"Unable to open {input_path}"
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

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 24.0

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    if not writer.isOpened():

        capture.release()

        raise RuntimeError(
            f"Unable to create {args.output}"
        )

    color_engine = (
        MathematicalColorField()
    )

    tone_engine = (
        MathematicalToneField()
    )

    frame_count = 0

    try:

        while True:

            ok, bgr = capture.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            # ------------------------------------------------
            # MTH-02
            # ------------------------------------------------

            color_frame = (
                color_engine.render(
                    rgb
                )
            )

            # ------------------------------------------------
            # MTH-03
            # ------------------------------------------------

            tone_frame = (
                tone_engine.render(
                    color_frame
                )
            )

            output_bgr = cv2.cvtColor(
                tone_frame,
                cv2.COLOR_RGB2BGR,
            )

            writer.write(
                output_bgr
            )

            frame_count += 1

            if frame_count % 24 == 0:

                print(
                    f"Processed "
                    f"{frame_count} frames"
                )

    finally:

        capture.release()
        writer.release()

    print(
        f"Completed: "
        f"{frame_count} frames"
    )

    print(
        f"Output: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
