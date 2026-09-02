"""Phase 6 Production Pipeline CLI Runner."""
import argparse
import sys
from pathlib import Path

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.project import Project
from src.core.pipeline import ProductionPipelineController


def main():
    parser = argparse.ArgumentParser(description="Animated Creator — Phase 6 Production Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input video (.mp4/.mov)")
    parser.add_argument("--style", "-s", type=str, default="anime_creator", help="Style key (anime_creator, clean_illustration, comic, watercolor, manga)")
    parser.add_argument("--project-dir", "-p", type=str, default=None, help="Directory for project workspace")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to process")
    parser.add_argument("--no-resume", action="store_true", help="Force recalculating all stages from scratch")

    args = parser.parse_args()

    input_p = Path(args.input)
    if not input_p.exists():
        print(f"[ERROR] Input video does not exist: {input_p}")
        sys.exit(1)

    proj_dir = Path(args.project_dir) if args.project_dir else Path("projects") / input_p.stem
    project = Project(proj_dir)
    controller = ProductionPipelineController(project=project, style_key=args.style)

    def on_progress(stage_name, pct):
        print(f"[*] [{pct:5.1f}%] {stage_name}...")

    final_master = controller.run(
        input_video_path=input_p,
        max_frames=args.max_frames,
        progress_callback=on_progress,
        resume=(not args.no_resume),
    )

    print("==================================================")
    print("         PROJECT EXECUTION COMPLETED!             ")
    print("==================================================")
    print(f"[+] Project Workspace: {project.root_dir.resolve()}")
    print(f"[+] Manifest:          {project.manifest_path.resolve()}")
    print(f"[+] YouTube Master:    {final_master}")
    print("==================================================")


if __name__ == "__main__":
    main()
