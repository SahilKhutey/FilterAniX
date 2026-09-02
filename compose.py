"""Multi-track final video composition CLI."""
import argparse
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.media.compose import VideoCompositor
from src.media.validate import OutputValidator


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Final Multi-Track Video Compositor")
    parser.add_argument("--video", "-v", type=str, required=True, help="Animated silent video (.mp4)")
    parser.add_argument("--audio-source", "-a", type=str, required=True, help="Original recording containing voice")
    parser.add_argument("--output", "-o", type=str, default="youtube_master.mp4", help="Master MP4 output path")
    parser.add_argument("--subtitles", "-s", type=str, default=None, help="Optional subtitles file (.srt)")
    parser.add_argument("--no-norm", action="store_true", help="Disable EBU R128 loudness normalization")

    args = parser.parse_args()

    print("==================================================")
    print("      PHASE 5: MULTI-TRACK MASTER COMPOSITOR      ")
    print("==================================================")
    print(f"[*] Animated Video: {args.video}")
    print(f"[*] Audio Source:   {args.audio_source}")
    print(f"[*] Subtitles:      {args.subtitles or 'None'}")
    print(f"[*] Loudness Target: {'Off' if args.no_norm else '-14 LUFS / -1.5 dBTP'}")
    print(f"[*] Output Target:  {args.output}")
    print("--------------------------------------------------")

    compositor = VideoCompositor()
    out_master = compositor.compose(
        video_path=args.video,
        audio_source_path=args.audio_source,
        output_path=args.output,
        subtitles_path=args.subtitles,
        normalize_loudness=(not args.no_norm),
    )

    print(f"[SUCCESS] YouTube Master MP4 generated: {out_master}")

    validator = OutputValidator()
    report = validator.validate(out_master)
    print(f"[+] Validation Status: {'[PASS]' if report.valid else '[FAIL]'}")
    if report.warnings:
        print(f"[!] Warnings: {report.warnings}")


if __name__ == "__main__":
    main()
