"""Diagnostic script to inspect vision detections on the creator fixture."""
from collections import Counter
from pathlib import Path
import json


VISION_FILE = Path("tests/fixtures/creator_vision.jsonl")
if not VISION_FILE.exists():
    VISION_FILE = Path("tests/fixtures/vision.jsonl")


def main() -> None:
    if not VISION_FILE.exists():
        print(f"[!] Vision JSONL file not found at {VISION_FILE}")
        return

    total = 0
    faces = 0
    poses = 0
    hands = 0

    blink_states = Counter()

    with VISION_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            total += 1

            if data.get("faces") or data.get("face"):
                faces += 1

            if data.get("pose"):
                poses += 1

            if data.get("hands"):
                hands += 1

            # Check blink states
            faces_data = data.get("faces")
            if faces_data and len(faces_data) > 0:
                f0 = faces_data[0]
                ear = (f0.get("left_eye_ear", 0.0) + f0.get("right_eye_ear", 0.0)) / 2.0
                state = "open" if ear > 0.15 else "closed"
                blink_states[state] += 1

    print()
    print("=== Synthetic Vision Diagnostics ===")
    print(f"Frames:       {total}")
    print(f"Face frames:  {faces}")
    print(f"Pose frames:  {poses}")
    print(f"Hand frames:  {hands}")
    print(f"Blink states: {dict(blink_states)}")

    if total:
        print()
        print(f"Face ratio: {faces / total:.3f}")
        print(f"Pose ratio: {poses / total:.3f}")
        print(f"Hand ratio: {hands / total:.3f}")


if __name__ == "__main__":
    main()
