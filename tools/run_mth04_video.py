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

from src.art.mathematical.palette_field import (
    MathematicalPaletteField,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="FilterAniX MTH-04 Video Runner (MTH-02 -> MTH-03 -> MTH-04)"
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="mth04_video.mp4",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames to render (default: full video)",
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

    palette_engine = (
        MathematicalPaletteField()
    )

    frame_count = 0

    print("=" * 64)
    print("FilterAniX MTH-04 Video Processing")
    print(f"Input:       {input_path}")
    print(f"Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"Output:      {args.output}")
    print("=" * 64)

    try:

        while True:

            if args.max_frames is not None and frame_count >= args.max_frames:
                break

            ok, bgr = capture.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB,
            )

            # MTH-02: Color Field
            mth02_res = color_engine.transform(rgb)

            # MTH-03: Tone Field
            mth03_res = tone_engine.transform(mth02_res.output_rgb)

            # MTH-04: Palette Field with MTH-03 Tone Preservation
            mth04_res = palette_engine.transform(
                mth03_res.output_rgb,
                tone_result=mth03_res,
            )

            output_bgr = cv2.cvtColor(
                mth04_res.output_rgb,
                cv2.COLOR_RGB2BGR,
            )

            writer.write(
                output_bgr
            )

            frame_count += 1

            if frame_count % 24 == 0 or frame_count == 1:

                print(
                    f"Rendered {frame_count} frames..."
                )

    finally:

        capture.release()
        writer.release()

    print("=" * 64)
    print(
        f"Completed {frame_count} frames -> {args.output}"
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
