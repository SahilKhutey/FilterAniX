"""Analyze a single image using the Phase 2 Vision Engine."""
import argparse
import json
import sys
from pathlib import Path
import cv2

# Ensure root dir in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision.vision_pipeline import VisionEngine


def analyze_image(image_path: str, output_dir: str = "image_output"):
    image_p = Path(image_path)
    out_dir_p = Path(output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if not image_p.exists():
        print(f"[ERROR] Image file not found: {image_p}")
        sys.exit(1)

    bgr = cv2.imread(str(image_p))
    if bgr is None:
        print(f"[ERROR] Failed to load image: {image_p}")
        sys.exit(1)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    print(f"[*] Analyzing image: {image_p} ({rgb.shape[1]}x{rgb.shape[0]})...")

    engine = VisionEngine()
    vision_data, annotated_rgb = engine.process_frame(rgb, frame_index=0, timestamp=0.0, generate_annotated=True)
    engine.close()

    # Save vision.json
    json_path = out_dir_p / "vision.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vision_data.to_dict(), f, indent=2)

    # Save annotated.jpg
    annotated_path = out_dir_p / "annotated.jpg"
    if annotated_rgb is not None:
        cv2.imwrite(str(annotated_path), cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

    print("==================================================")
    print("           VISION ANALYSIS RESULTS                ")
    print("==================================================")
    print(f"[+] Faces Detected:  {len(vision_data.faces)}")
    if vision_data.faces:
        f = vision_data.faces[0]
        print(f"    - Face #0 Landmarks: {f.landmark_count}")
        print(f"    - Mouth Opening:     {f.mouth_opening:.3f}")
        print(f"    - Left/Right EAR:    {f.left_eye_ear:.3f} / {f.right_eye_ear:.3f}")
    print(f"[+] Body Pose:       {'Detected' if vision_data.pose else 'None'}")
    print(f"[+] Hands Detected:  {len(vision_data.hands)}")
    for h in vision_data.hands:
        print(f"    - Hand: {h.label} (Conf: {h.confidence:.2f})")
    print(f"[+] Person Mask:     {vision_data.person_mask.coverage*100:.1f}% coverage" if vision_data.person_mask else "[+] Person Mask: None")
    print(f"[+] Objects:         {[obj.label for obj in vision_data.objects]}")
    print("--------------------------------------------------")
    print(f"[SUCCESS] Saved structured JSON to:   {json_path}")
    print(f"[SUCCESS] Saved annotated preview to: {annotated_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Single Image Vision Analyzer")
    parser.add_argument("image", type=str, help="Path to input image (JPG/PNG)")
    parser.add_argument("--output-dir", "-o", type=str, default="image_output", help="Directory for output files")
    args = parser.parse_args()

    analyze_image(args.image, args.output_dir)


if __name__ == "__main__":
    main()
