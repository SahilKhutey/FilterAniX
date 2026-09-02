"""Analyze full video consistency against a reference profile and export report."""
import argparse
import sys
from pathlib import Path
import json

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.consistency.types import ReferenceProfile
from src.consistency.report import ConsistencyAuditor


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Video Consistency & Quality Auditor")
    parser.add_argument("--video", "-v", type=str, required=True, help="Path to animated video (.mp4)")
    parser.add_argument("--reference-profile", "-r", type=str, required=True, help="Path to reference_profile.json")
    parser.add_argument("--output", "-o", type=str, default="consistency_report.json", help="Path for report JSON")
    parser.add_argument("--warning-threshold", "-w", type=float, default=0.55, help="Warning threshold")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to evaluate")

    args = parser.parse_args()

    prof_p = Path(args.reference_profile)
    if not prof_p.exists():
        print(f"[ERROR] Reference profile not found: {prof_p}")
        sys.exit(1)

    with open(prof_p, "r", encoding="utf-8") as f:
        profile = ReferenceProfile.from_dict(json.load(f))

    auditor = ConsistencyAuditor(profile, warning_threshold=args.warning_threshold)
    report = auditor.audit_video(
        video_path=args.video,
        output_report_path=args.output,
        max_frames=args.max_frames,
    )

    print("==================================================")
    print("          VIDEO CONSISTENCY AUDIT REPORT          ")
    print("==================================================")
    print(f"[+] Total Frames:        {report.frames} ({report.duration_seconds:.2f}s @ {report.fps:.1f} FPS)")
    print(f"[+] Mean Similarity:     {report.mean_similarity:.4f} ({report.mean_similarity*100:.1f}%)")
    print(f"[+] Min / Max Score:     {report.minimum_similarity:.4f} / {report.maximum_similarity:.4f}")
    print(f"[+] Warning Frames:      {report.warning_frame_count} ({report.warning_ratio*100:.2f}%)")
    print("--------------------------------------------------")
    print(f"[SUCCESS] Consistency report saved to: {args.output}")


if __name__ == "__main__":
    main()
