"""Precompute temporal decisions and export temporal_plan.jsonl."""
import argparse
import sys
from pathlib import Path
import json

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.consistency.types import ReferenceProfile
from src.consistency.planner import TemporalPlanner


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Temporal Plan Generator")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to input video (.mp4)")
    parser.add_argument("--vision", "--vision-jsonl", "-j", dest="vision_jsonl", type=str, default=None, help="Path to vision.jsonl")
    parser.add_argument("--output", "-o", type=str, default="temporal_plan.jsonl", help="Output plan path")
    parser.add_argument("--reference-profile", "-r", type=str, default=None, help="Optional reference profile")
    parser.add_argument("--keyframe-interval", "-k", type=int, default=12, help="Anchor keyframe interval")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to plan")

    args = parser.parse_args()

    profile = None
    if args.reference_profile and Path(args.reference_profile).exists():
        with open(args.reference_profile, "r", encoding="utf-8") as f:
            profile = ReferenceProfile.from_dict(json.load(f))

    print("==================================================")
    print("          BUILDING TEMPORAL RENDER PLAN           ")
    print("==================================================")
    print(f"[*] Input Video:        {args.video}")
    print(f"[*] Keyframe Interval:  Every {args.keyframe_interval} frames")
    print(f"[*] Vision Data:        {args.vision_jsonl or 'None'}")
    print(f"[*] Output Plan:        {args.output}")
    print("--------------------------------------------------")

    planner = TemporalPlanner(keyframe_interval=args.keyframe_interval, reference_profile=profile)
    out_path = planner.generate_plan(
        video_path=args.video,
        vision_jsonl_path=args.vision_jsonl,
        output_plan_path=args.output,
        max_frames=args.max_frames,
    )

    print(f"[SUCCESS] Temporal render plan saved to: {out_path}")


if __name__ == "__main__":
    main()
