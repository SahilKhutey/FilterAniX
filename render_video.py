"""Phase 3 Full Video Stylization CLI Runner."""
import argparse
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.art.video_renderer import VideoStyleRenderer
from src.art.types import RenderConfig, StylePreset


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Video Artistic Style Engine")
    parser.add_argument("video", type=str, help="Path to input video (.mp4/.mov)")
    parser.add_argument("--vision-jsonl", "-j", type=str, default=None, help="Path to Phase 2 vision.jsonl file")
    parser.add_argument("--reference", "-r", type=str, default=None, help="Optional character reference image path")
    parser.add_argument("--output", "-o", type=str, default="animated_preview.mp4", help="Output animated video path")
    parser.add_argument("--side-by-side", action="store_true", help="Render side-by-side comparison video")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Optional frame limit")

    args = parser.parse_args()

    print("==================================================")
    print("      PHASE 3: ARTISTIC STYLE VIDEO ENGINE        ")
    print("==================================================")
    print(f"[*] Input Video:   {args.video}")
    print(f"[*] Vision Data:   {args.vision_jsonl or 'Auto (procedural vision)'}")
    print(f"[*] Reference Img: {args.reference or 'None'}")
    print(f"[*] Output Target: {args.output}")
    print("--------------------------------------------------")

    config = RenderConfig(reference_image_path=args.reference)
    renderer = VideoStyleRenderer(config)

    def on_progress(p):
        print(f"\r[*] {p.status_message} (Speed: {p.fps:.1f} FPS, ETA: {p.eta_sec:.1f}s)", end="", flush=True)

    final_out = renderer.render_video(
        input_path=args.video,
        output_path=args.output,
        vision_jsonl=args.vision_jsonl,
        reference_image_path=args.reference,
        max_frames=args.max_frames,
        side_by_side=args.side_by_side,
        progress_callback=on_progress,
    )
    print(f"\n[SUCCESS] Render complete! Final animated video with synced audio: {final_out}")


if __name__ == "__main__":
    main()
