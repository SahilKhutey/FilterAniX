import argparse
import json
from pathlib import Path
import cv2

from .preprocess import make_edge_control, make_pose_control


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--vision", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = cv2.imread(str(args.input))
    if frame is None:
        raise RuntimeError("Could not read input.")

    if args.vision:
        vision = json.loads(args.vision.read_text(encoding="utf-8"))
        control = make_pose_control(frame, vision)
    else:
        control = make_edge_control(frame)

    cv2.imwrite(str(args.output), control)


if __name__ == "__main__":
    main()
