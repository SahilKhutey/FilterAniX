"""YouTube Multi-Resolution Exporter CLI."""
import argparse
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.export import YouTubeExporter, YOUTUBE_PRESETS


def main():
    parser = argparse.ArgumentParser(description="Phase 6 YouTube Master Exporter")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to YouTube master MP4")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path for exported MP4")
    parser.add_argument("--preset", "-p", type=str, default="1080p", choices=list(YOUTUBE_PRESETS.keys()), help="Target preset")

    args = parser.parse_args()

    print("==================================================")
    print("           YOUTUBE MULTI-RESOLUTION EXPORT        ")
    print("==================================================")
    print(f"[*] Input Master: {args.input}")
    print(f"[*] Target Preset:{args.preset}")
    print(f"[*] Output Path:  {args.output}")
    print("--------------------------------------------------")

    exporter = YouTubeExporter(preset_name=args.preset)
    out_file = exporter.export(input_master_path=args.input, output_export_path=args.output)

    print(f"[SUCCESS] Exported YouTube {args.preset} Master to: {out_file}")


if __name__ == "__main__":
    main()
