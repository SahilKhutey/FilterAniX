"""Analyze single image consistency against a reference profile."""
import argparse
import json
import sys
from pathlib import Path
import cv2

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.consistency.types import ReferenceProfile
from src.consistency.identity import IdentityScorer


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Single Image Consistency Evaluator")
    parser.add_argument("image", type=str, help="Path to target image frame")
    parser.add_argument("--reference-profile", "-r", type=str, required=True, help="Path to reference_profile.json")
    parser.add_argument("--warning-threshold", "-w", type=float, default=0.55, help="Threshold below which warning is raised")

    args = parser.parse_args()

    img_p = Path(args.image)
    prof_p = Path(args.reference_profile)

    if not img_p.exists() or not prof_p.exists():
        print("[ERROR] Input image or reference profile does not exist.")
        sys.exit(1)

    with open(prof_p, "r", encoding="utf-8") as f:
        profile = ReferenceProfile.from_dict(json.load(f))

    bgr = cv2.imread(str(img_p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    scorer = IdentityScorer(profile, warning_threshold=args.warning_threshold)
    metrics = scorer.evaluate_frame(rgb)

    print("==================================================")
    print("           FRAME CONSISTENCY EVALUATION           ")
    print("==================================================")
    print(f"[*] Overall Similarity:   {metrics.similarity:.4f} ({metrics.similarity*100:.1f}%)")
    print(f"[*] Color Similarity:     {metrics.color_similarity:.4f}")
    print(f"[*] Edge Similarity:      {metrics.edge_similarity:.4f}")
    print(f"[*] Quality Warning:      {'[WARNING] Low similarity' if metrics.warning else '[PASS] Consistent'}")
    print("==================================================")


if __name__ == "__main__":
    main()
