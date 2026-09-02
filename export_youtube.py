from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from src.media.ffmpeg import require_ffmpeg


PRESETS = {
    "720p": {
        "scale": "1280:720",
        "crf": "20",
    },
    "1080p": {
        "scale": "1920:1080",
        "crf": "18",
    },
    "1440p": {
        "scale": "2560:1440",
        "crf": "17",
    },
    "2160p": {
        "scale": "3840:2160",
        "crf": "16",
    },
}


def export(
    input_path: str | Path,
    output_path: str | Path,
    preset: str = "1080p",
) -> Path:
    """Encodes master video using YouTube-recommended bitrates, scaling, and faststart MP4 flags."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")

    settings = PRESETS[preset]
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = require_ffmpeg()

    command = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-vf", (
            f"scale={settings['scale']}:"
            "force_original_aspect_ratio=decrease,"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        ),
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", settings["crf"],
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-movflags", "+faststart",
        str(out_p),
    ]

    subprocess.run(command, check=True)
    return out_p


def main():
    parser = argparse.ArgumentParser(description="Export YouTube-ready Master Video")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preset", choices=PRESETS.keys(), default="1080p")

    args = parser.parse_args()
    export(args.input, args.output, args.preset)
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()
