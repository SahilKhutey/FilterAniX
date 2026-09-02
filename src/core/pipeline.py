from __future__ import annotations

from pathlib import Path
import json
import shutil
import traceback
from typing import Any, Dict, Optional

from src.core.project import Project


class PipelineManager:
    """Orchestrates Phases 1 through 5 in a resilient, resumable production pipeline."""

    def __init__(self, project: Project):
        self.project = project

    def run_stage(
        self,
        name: str,
        function,
        *args,
        **kwargs,
    ):
        if self.project.stage_complete(name):
            print(f"[SKIP] {name} already complete")
            return self.project.load()["stages"][name]["output"]

        self.project.update_stage(name, "running")
        try:
            print(f"[START] {name}")
            result = function(*args, **kwargs)
            self.project.update_stage(name, "complete", output=result)
            print(f"[DONE] {name}")
            return result
        except Exception as exc:
            self.project.update_stage(name, "failed", error=str(exc))
            traceback.print_exc()
            raise

    def run(self, input_video: str | Path, style: str = "anime_creator") -> Dict[str, Any]:
        project_root = self.project.root
        source = Path(input_video).resolve()

        if not source.exists():
            raise FileNotFoundError(f"Source video not found: {source}")

        # PHASE 1: Video Input & Processing
        phase1_output = self.run_stage(
            "input",
            self._phase1,
            source,
            project_root,
        )

        # PHASE 2: Vision & Scene Understanding
        vision_output = self.run_stage(
            "vision",
            self._phase2,
            phase1_output,
            project_root,
        )

        # PHASE 3: Artistic Style Engine
        artistic_output = self.run_stage(
            "artistic",
            self._phase3,
            phase1_output,
            vision_output,
            style,
            project_root,
        )

        # PHASE 4: Character Consistency & Temporal Planning
        consistency_output = self.run_stage(
            "consistency",
            self._phase4,
            phase1_output,
            vision_output,
            artistic_output,
            project_root,
        )

        # PHASE 5A: Lip-Sync Timeline
        lipsync_output = self.run_stage(
            "lipsync",
            self._phase5_lipsync,
            phase1_output,
            vision_output,
            project_root,
        )

        # PHASE 5B: Media Composition & Multiplexing
        final_output = self.run_stage(
            "composition",
            self._phase5_compose,
            artistic_output,
            phase1_output,
            lipsync_output,
            consistency_output,
            project_root,
        )

        # VALIDATION
        validation_output = self.run_stage(
            "validation",
            self._validate,
            final_output,
            project_root,
        )

        return {
            "final_video": str(final_output),
            "validation": str(validation_output),
        }

    # ------------------------------------
    # Phase adapters
    # ------------------------------------

    def _phase1(self, source: Path, root: Path) -> Path:
        output = root / "source" / source.name
        if source != output:
            shutil.copy2(str(source), str(output))
        return output

    def _phase2(self, source: Path, root: Path) -> Path:
        from src.vision.video_analyzer import analyze_video
        from src.vision.engine import VisionEngine
        from src.vision.jsonl import write_vision_jsonl

        output = root / "vision" / "vision.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)

        try:
            analyze_video(str(source), str(output))
        except Exception:
            engine = VisionEngine()
            frames_data = engine.process_video(source)
            write_vision_jsonl(frames_data, str(output))

        return output

    def _phase3(
        self,
        source: Path,
        vision: Path,
        style: str,
        root: Path,
    ) -> Path:
        from src.art.video_renderer import VideoRenderer
        from src.art.types import StyleConfig

        output = root / "artistic" / "animated.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        config = StyleConfig(name=str(style))
        renderer = VideoRenderer(config=config)
        renderer.render_video(
            video_path=str(source),
            vision_jsonl_path=str(vision),
            output_path=str(output),
        )
        return output


    def _phase4(
        self,
        source: Path,
        vision: Path,
        artistic: Path,
        root: Path,
    ) -> Path:
        from src.consistency.planner import TemporalPlanner

        output = root / "consistency" / "temporal_plan.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)

        planner = TemporalPlanner(keyframe_interval=12)
        planner.generate_plan(
            video_path=str(source),
            output_plan_path=str(output),
        )
        return output

    def _phase5_lipsync(
        self,
        source: Path,
        vision: Path,
        root: Path,
    ) -> Path:
        from build_lipsync import build_lipsync

        output = root / "lipsync" / "lipsync.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)

        build_lipsync(
            video=str(source),
            vision_jsonl=str(vision),
            output=str(output),
        )
        return output

    def _phase5_compose(
        self,
        artistic: Path,
        source: Path,
        lipsync: Path,
        consistency: Path,
        root: Path,
    ) -> Path:
        from src.media.compose import compose_final_video

        output = root / "output" / "youtube_master.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        compose_final_video(
            animated_video=str(artistic),
            audio_source=str(source),
            output=str(output),
        )
        return output

    def _validate(self, video: Path, root: Path) -> Path:
        from src.media.validate import validate_video

        result = validate_video(str(video))
        report = root / "reports" / "validation.json"
        report.parent.mkdir(parents=True, exist_ok=True)

        with open(str(report), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if not result.get("valid", True):
            raise RuntimeError(f"Final video validation failed: {result.get('errors')}")

        return report
