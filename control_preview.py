"""Inspect and export structural control maps (edge, pose, face, hands)."""
import argparse
import json
import sys
from pathlib import Path
import cv2

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.art.preprocess import ControlBuilder
from src.vision.models import FrameVisionData, FaceData, PoseData, HandData, Landmark, BoundingBox


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Structural Control Preview")
    parser.add_argument("input", type=str, help="Path to input image (JPG/PNG)")
    parser.add_argument("--output", "-o", type=str, default="control_preview.jpg", help="Path for control preview output")
    parser.add_argument("--vision", "-v", type=str, default=None, help="Optional vision.json data path")

    args = parser.parse_args()

    image_p = Path(args.input)
    output_p = Path(args.output)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not image_p.exists():
        print(f"[ERROR] Image file not found: {image_p}")
        sys.exit(1)

    bgr = cv2.imread(str(image_p))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    vision_data = None
    if args.vision and Path(args.vision).exists():
        with open(args.vision, "r", encoding="utf-8") as f:
            d = json.load(f)
            faces = [
                FaceData(
                    face_id=f_d.get("face_id", 0),
                    landmarks=[Landmark(**lm) for lm in f_d.get("landmarks", [])],
                    bbox=BoundingBox(**f_d["bbox"]),
                    landmark_count=f_d.get("landmark_count", 0),
                )
                for f_d in d.get("faces", [])
            ]
            pose = None
            if d.get("pose"):
                pose = PoseData(
                    landmarks=[Landmark(**lm) for lm in d["pose"].get("landmarks", [])],
                    bbox=BoundingBox(**d["pose"]["bbox"]),
                    landmark_count=d["pose"].get("landmark_count", 0),
                )
            hands = [
                HandData(
                    label=h_d.get("label", "Unknown"),
                    confidence=h_d.get("confidence", 0.9),
                    landmarks=[Landmark(**lm) for lm in h_d.get("landmarks", [])],
                    bbox=BoundingBox(**h_d["bbox"]),
                )
                for h_d in d.get("hands", [])
            ]
            vision_data = FrameVisionData(
                frame_index=0, timestamp=0.0, width=rgb.shape[1], height=rgb.shape[0],
                faces=faces, pose=pose, hands=hands
            )

    builder = ControlBuilder()
    control_map = builder.build_control_map(rgb, vision_data)

    cv2.imwrite(str(output_p), cv2.cvtColor(control_map.combined_control, cv2.COLOR_RGB2BGR))
    print(f"[SUCCESS] Exported structural control map to: {output_p}")


if __name__ == "__main__":
    main()
