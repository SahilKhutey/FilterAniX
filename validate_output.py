"""Validate final master video output and A/V duration alignment."""
import argparse
import json
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.media.validate import OutputValidator


def main():
    parser = argparse.ArgumentParser(description="Phase 5 YouTube Master Validator")
    parser.add_argument("media", type=str, help="Path to final output MP4")
    parser.add_argument("--output-json", "-o", type=str, default=None, help="Optional path to export validation JSON")
    args = parser.parse_args()

    validator = OutputValidator()
    result = validator.validate(args.media)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    print("==================================================")
    print("           YOUTUBE MASTER VALIDATION              ")
    print("==================================================")
    print(f"[*] File:            {Path(result.path).name}")
    print(f"[*] Resolution:      {result.resolution} @ {result.fps:.2f} FPS")
    print(f"[*] Video Duration:  {result.video_duration:.3f}s")
    print(f"[*] Audio Duration:  {result.audio_duration:.3f}s")
    print(f"[*] A/V Drift:       {result.drift_seconds:.3f}s")
    print(f"[*] Container Health:{'[PASS] Ready for YouTube' if result.valid else '[FAIL]'}")
    if result.warnings:
        for w in result.warnings:
            print(f"    [!] {w}")
    print("==================================================")


if __name__ == "__main__":
    main()
