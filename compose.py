import argparse
from src.media.compose import compose_final_video


def main():
    parser = argparse.ArgumentParser(
        description="Compose final Animated Creator video."
    )

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--audio-source",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--subtitles",
        default=None,
    )

    args = parser.parse_args()

    output = compose_final_video(
        animated_video=args.video,
        audio_source=args.audio_source,
        output=args.output,
        subtitles=args.subtitles,
    )

    print(f"Final video created: {output}")


if __name__ == "__main__":
    main()
