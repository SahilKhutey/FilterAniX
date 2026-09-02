"""FilterAniX Prototype v0.1 CLI Runner."""
import argparse
import sys
from pathlib import Path

# Ensure root directory and archive directory are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

from filteranix.core.config import load_config
from filteranix.pipeline.offline_pipeline import OfflineVideoPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FilterAniX - Video-to-Anime Transformation Engine (Prototype v0.1)"
    )
    parser.add_argument(
        "--input", "-i", type=str, default=None, help="Path to input real-world creator video (.mp4/.mov)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output_anime.mp4", help="Path for output stylized video"
    )
    parser.add_argument(
        "--pipeline-config", "-p", type=str, default="configs/default_pipeline.yaml", help="Pipeline configuration YAML"
    )
    parser.add_argument(
        "--style", "-s", type=str, default="configs/styles/creator_anime.yaml", help="Style profile YAML"
    )
    parser.add_argument(
        "--character", "-c", type=str, default="configs/characters/creator_default.yaml", help="Character identity YAML"
    )
    parser.add_argument(
        "--side-by-side", action="store_true", help="Render side-by-side comparison (Real Video | Anime Video)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional maximum frames to process"
    )
    parser.add_argument(
        "--synthetic-demo", action="store_true", help="Generate and process a synthetic test video"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"==================================================")
    print(f"    FilterAniX: Video-to-Anime Engine v0.1        ")
    print(f"==================================================")
    print(f"[*] Loading configuration...")
    config = load_config(
        pipeline_path=args.pipeline_config if Path(args.pipeline_config).exists() else None,
        style_path=args.style if Path(args.style).exists() else None,
        character_path=args.character if Path(args.character).exists() else None,
    )
    print(f"[+] Loaded Style: '{config.style.name}'")
    print(f"[+] Loaded Character: '{config.character.name}'")

    input_file = args.input
    if args.synthetic_demo or not input_file:
        from tests.test_archive_offline_pipeline_smoke import generate_synthetic_creator_video
        synthetic_path = Path("samples/synthetic_creator_test.mp4")
        if not synthetic_path.exists() or args.synthetic_demo:
            print(f"[*] Generating synthetic 60-frame creator test scene at {synthetic_path}...")
            generate_synthetic_creator_video(synthetic_path, num_frames=60, width=640, height=360)
        input_file = str(synthetic_path)
        print(f"[*] Using synthetic input: {input_file}")

    pipeline = OfflineVideoPipeline(config)

    print(f"[*] Processing: {input_file} -> {args.output}")
    pipeline.process_video(
        input_path=input_file,
        output_path=args.output,
        side_by_side=args.side_by_side,
        max_frames=args.max_frames,
    )
    print(f"[+] Done! Process completed successfully.")


if __name__ == "__main__":
    main()
