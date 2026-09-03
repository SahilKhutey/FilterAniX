import argparse
from pathlib import Path

from src.core.project import Project
from src.core.pipeline import PipelineManager
from src.core.recovery import recover_project
from src.core.logging_setup import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Animated Creator Production Pipeline")
    parser.add_argument("--input", required=True, help="Path to input creator video")
    parser.add_argument("--project", "--project-dir", dest="project", required=True, help="Path to project directory")
    parser.add_argument("--style", default="anime_creator", help="Style key (e.g. anime_creator, illustration)")

    args = parser.parse_args()

    logger = setup_logging()
    project_path = Path(args.project)
    project = Project(project_path)

    if not project.manifest_path.exists():
        project.create(project_path.name)

    recover_project(project)
    pipeline = PipelineManager(project)

    logger.info("Starting Animated Creator pipeline")
    result = pipeline.run(
        input_video=args.input,
        style=args.style,
    )

    logger.info("Pipeline completed: %s", result)
    print()
    print("FINAL VIDEO:")
    print(result["final_video"])


if __name__ == "__main__":
    main()
