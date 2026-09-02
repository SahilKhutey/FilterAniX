from __future__ import annotations

import argparse

from src.art.types import StyleConfig
from src.art.video_renderer import VideoRenderer


def main():
    parser = argparse.ArgumentParser(
        description="Render Animated Creator artistic video"
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--vision-jsonl",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--style",
        default="anime_creator",
    )

    args = parser.parse_args()

    config = StyleConfig(
        name=args.style
    )

    renderer = VideoRenderer(config)

    result = renderer.render(
        input_video=args.video,
        vision_jsonl=args.vision_jsonl,
        output_video=args.output,
    )

    print()
    print("=== ARTISTIC RENDER COMPLETE ===")
    print(f"Frames : {result['frames']}")
    print(f"FPS    : {result['fps']:.2f}")
    print(
        f"Size   : "
        f"{result['width']}x{result['height']}"
    )
    print(f"Output : {result['output']}")


if __name__ == "__main__":
    main()
