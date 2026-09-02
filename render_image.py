"""Render a single image into stylized anime illustration."""
import argparse
import json
import sys
from pathlib import Path
import cv2

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.art.style_engine import StyleEngine
from src.art.types import RenderConfig, StylePreset
from src.vision.models import FrameVisionData, FaceData, PoseData, HandData, Landmark, BoundingBox


def render_image(
    image_path: str,
    output_path: str = "output.jpg",
    reference_path: str = None,
    vision_path: str = None,
):
    image_p = Path(image_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if not image_p.exists():
        print(f"[ERROR] Input image not found: {image_p}")
        sys.exit(1)

    bgr = cv2.imread(str(image_p))
    if bgr is None:
        print(f"[ERROR] Failed to read image: {image_p}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    ref_rgb = None
    if reference_path and Path(reference_path).exists():
        ref_bgr = cv2.imread(str(reference_path))
        if ref_bgr is not None:
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            print(f"[+] Loaded Reference Character Palette: {reference_path}")

    vision_data = None
    if vision_path and Path(vision_path).exists():
        with open(vision_path, "r", encoding="utf-8") as f:
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
                frame_index=d.get("frame_index", 0),
                timestamp=d.get("timestamp", 0.0),
                width=d.get("width", rgb.shape[1]),
                height=d.get("height", rgb.shape[0]),
                faces=faces,
                pose=pose,
                hands=hands,
            )
            print(f"[+] Loaded Vision Guidance Data: {vision_path}")

    print(f"[*] Rendering image through Phase 3 Style Engine...")
    engine = StyleEngine()
    art_rgb = engine.render_frame(rgb, vision_data=vision_data, reference_rgb=ref_rgb, stabilize=False)

    cv2.imwrite(str(output_p), cv2.cvtColor(art_rgb, cv2.COLOR_RGB2BGR))
    print(f"[SUCCESS] Exported stylized image to: {output_p}")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Image Stylization Tool")
    parser.add_argument("input", type=str, help="Input image path (JPG/PNG)")
    parser.add_argument("--output", "-o", type=str, default="output.jpg", help="Output stylized image path")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Optional character reference image path")
    parser.add_argument("--vision", "-v", type=str, default=None, help="Optional vision.json file path")

    args = parser.parse_args()
    render_image(args.input, args.output, args.reference, args.vision)


if __name__ == "__main__":
    main()
