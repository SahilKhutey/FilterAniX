import argparse
import json
from src.media.validate import probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    args = parser.parse_args()

    data = probe(args.video)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
