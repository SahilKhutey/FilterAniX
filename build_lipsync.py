"""Extract and smooth creator lip-sync timeline into lipsync.jsonl."""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.lipsync.analyzer import LipSyncAnalyzer, LipSyncRecord
from src.lipsync.smoother import LipSyncSmoother
from src.vision.models import FrameVisionData, FaceData, Landmark, BoundingBox


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Lip-Sync Timeline Extractor")
    parser.add_argument("--video", "-v", type=str, required=True, help="Input video path")
    parser.add_argument("--vision-jsonl", "-j", type=str, required=True, help="Path to Phase 2 vision.jsonl")
    parser.add_argument("--output", "-o", type=str, default="lipsync.jsonl", help="Output lipsync.jsonl path")
    parser.add_argument("--window-size", "-w", type=int, default=5, help="Temporal smoothing window size")

    args = parser.parse_args()

    jsonl_p = Path(args.vision_jsonl)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not jsonl_p.exists():
        print(f"[ERROR] Vision data file not found: {jsonl_p}")
        sys.exit(1)

    print("==================================================")
    print("       PHASE 5: LIP-SYNC TIMELINE GENERATOR       ")
    print("==================================================")
    print(f"[*] Reading Vision Data: {jsonl_p}")
    print(f"[*] Smoothing Window:    {args.window_size} frames")
    print(f"[*] Output Target:       {out_p}")
    print("--------------------------------------------------")

    analyzer = LipSyncAnalyzer()
    raw_records = []

    with open(jsonl_p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            idx = d.get("frame_index", 0)
            ts = d.get("timestamp", 0.0)

            faces = []
            for f_d in d.get("faces", []):
                faces.append(
                    FaceData(
                        face_id=f_d.get("face_id", 0),
                        landmarks=[],
                        bbox=BoundingBox(**f_d["bbox"]),
                        landmark_count=f_d.get("landmark_count", 0),
                        mouth_opening=f_d.get("mouth_opening", 0.0),
                    )
                )

            vision_data = FrameVisionData(
                frame_index=idx, timestamp=ts, width=1920, height=1080, faces=faces
            )
            rec = analyzer.analyze_frame(frame_index=idx, timestamp=ts, vision_data=vision_data)
            raw_records.append(rec)

    smoother = LipSyncSmoother(window_size=args.window_size)
    smoothed_records = smoother.smooth_timeline(raw_records)

    with open(out_p, "w", encoding="utf-8") as out_f:
        for r in smoothed_records:
            out_f.write(json.dumps(r.to_dict()) + "\n")

    print(f"[SUCCESS] Generated {len(smoothed_records)} smoothed viseme records in: {out_p}")


if __name__ == "__main__":
    main()
