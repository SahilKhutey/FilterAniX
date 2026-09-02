"""Build canonical character reference profile JSON from an image."""
import argparse
import json
import sys
from pathlib import Path
import cv2

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.consistency.identity import IdentityProfileBuilder


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Character Reference Profile Builder")
    parser.add_argument("reference", type=str, help="Path to character reference image (JPG/PNG)")
    parser.add_argument("--output", "-o", type=str, default="reference_profile.json", help="Output profile JSON path")
    parser.add_argument("--name", "-n", type=str, default="creator_canonical", help="Character name / ID")

    args = parser.parse_args()

    ref_p = Path(args.reference)
    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not ref_p.exists():
        print(f"[ERROR] Reference image not found: {ref_p}")
        sys.exit(1)

    bgr = cv2.imread(str(ref_p))
    if bgr is None:
        print(f"[ERROR] Failed to load reference image: {ref_p}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[*] Extracting visual signature from: {ref_p} ({rgb.shape[1]}x{rgb.shape[0]})...")

    profile = IdentityProfileBuilder.build_profile(rgb, name=args.name)

    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(profile.to_dict(), f, indent=2)

    print("==================================================")
    print("      CHARACTER REFERENCE PROFILE GENERATED       ")
    print("==================================================")
    print(f"[+] Character Name:   {profile.name}")
    print(f"[+] Aspect Ratio:     {profile.aspect_ratio:.2f}")
    print(f"[+] Edge Density:     {profile.edge_density*100:.2f}%")
    print(f"[+] Dominant Colors:  {len(profile.dominant_palette)} palette centroids")
    print("--------------------------------------------------")
    print(f"[SUCCESS] Saved reference profile to: {out_p}")


if __name__ == "__main__":
    main()
