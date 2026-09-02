"""Inspect media container details using FFmpeg."""
import argparse
import json
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.media.ffmpeg import inspect_media


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Media Stream Inspector")
    parser.add_argument("input", type=str, help="Path to video/audio media container")
    args = parser.parse_args()

    media_p = Path(args.input)
    if not media_p.exists():
        print(f"[ERROR] File not found: {media_p}")
        sys.exit(1)

    details = inspect_media(media_p)

    print("==================================================")
    print("           MEDIA STREAM INSPECTION                ")
    print("==================================================")
    print(f"[*] File Path:       {details.path}")
    print(f"[*] Video Track:     {'Yes' if details.has_video else 'No'} ({details.video_codec})")
    print(f"[*] Resolution:      {details.resolution_str} @ {details.fps:.2f} FPS")
    print(f"[*] Video Duration:  {details.video_duration:.3f}s")
    print("--------------------------------------------------")
    print(f"[*] Audio Track:     {'Yes' if details.has_audio else 'No'} ({details.audio_codec})")
    print(f"[*] Audio Details:   {details.sample_rate} Hz, {details.channels} Channels")
    print(f"[*] Audio Duration:  {details.audio_duration:.3f}s")
    print("==================================================")


if __name__ == "__main__":
    main()
