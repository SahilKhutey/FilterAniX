from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2

from src.lipsync.analyzer import analyze_mouth_frame
from src.lipsync.smoother import smooth_timeline


def build_lipsync(
    video: str | Path,
    vision_jsonl: str | Path,
    output: str | Path,
) -> Path:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    vision_frames = {}
    with open(str(vision_jsonl), "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            index = int(data.get("frame_index", data.get("frame", 0)))
            vision_frames[index] = data

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for index in range(total_frames):
        observation = vision_frames.get(index, {})

        face = observation.get("face", {})
        if isinstance(face, list):
            face = face[0] if face else {}

        mouth = (
            face.get("mouth")
            if isinstance(face, dict) and face.get("mouth")
            else face
        )

        result = analyze_mouth_frame(
            frame_index=index,
            timestamp=index / fps,
            observation=mouth,
        )
        frames.append(result)

    cap.release()

    frames = smooth_timeline(frames, window=3)

    out_p = Path(output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(str(out_p), "w", encoding="utf-8") as f:
        for frame in frames:
            f.write(json.dumps(frame.to_dict()) + "\n")

    return out_p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--vision-jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_p = build_lipsync(args.video, args.vision_jsonl, args.output)
    print(f"Created lip-sync timeline: {out_p}")


if __name__ == "__main__":
    main()
