import argparse
import json
from src.media.validate import validate_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    result = validate_video(args.video)
    print(json.dumps(result, indent=2))

    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
