from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2

from src.art.mathematical import (
    MathematicalColorField,
    MathematicalToneField,
    MathematicalPaletteField,
    MathematicalEdgeField,
    MathematicalShadowHighlightField,
)


def main() -> None:

    parser = argparse.ArgumentParser(
        description="FilterAniX MTH-06 Video Runner (MTH-02 -> MTH-06)"
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
        help="Optional limit on number of frames to process",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to create {output_path}")

    color_engine = MathematicalColorField()
    tone_engine = MathematicalToneField()
    palette_engine = MathematicalPaletteField()
    edge_engine = MathematicalEdgeField()
    lighting_engine = MathematicalShadowHighlightField()

    frame_index = 0

    print("=" * 64)
    print("FilterAniX MTH-06 Video Processing (MTH-02 -> MTH-06)")
    print(f"Input:       {input_path}")
    print(f"Resolution:  {width}x{height} @ {fps:.1f} FPS")
    print(f"Output:      {output_path}")
    print("=" * 64)

    try:
        while True:
            if args.max_frames is not None and frame_index >= args.max_frames:
                break

            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # MTH-02 Color Field
            mth02 = color_engine.transform(rgb)

            # MTH-03 Tone Field
            mth03 = tone_engine.transform(mth02.output_rgb)

            # MTH-04 Palette Field
            mth04 = palette_engine.transform(
                mth03.output_rgb,
                tone_result=mth03,
            )

            # MTH-05 Edge Field
            mth05 = edge_engine.transform(mth04.output_rgb)

            # MTH-06 Shadow/Highlight Field
            mth06 = lighting_engine.transform(mth05.output_rgb)

            output_bgr = cv2.cvtColor(mth06.output_rgb, cv2.COLOR_RGB2BGR)
            writer.write(output_bgr)

            frame_index += 1
            if frame_index % 24 == 0 or frame_index == 1:
                total_str = f"/{frame_count}" if frame_count > 0 else ""
                print(f"Rendered {frame_index}{total_str} frames...")

    finally:
        cap.release()
        writer.release()

    print()
    print(f"Processed frames: {frame_index}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
