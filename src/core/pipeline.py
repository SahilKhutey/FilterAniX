from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.cancellation import JobCancelledError, JobControl
from src.core.project import Project
from src.core.project_lock import ProjectLock
from src.core.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)

STAGE_WEIGHTS = {
    "input": 0.05,
    "vision": 0.15,
    "consistency": 0.10,
    "lipsync": 0.05,
    "artistic": 0.50,
    "composition": 0.10,
    "validation": 0.05,
}


def calculate_progress(
    completed_weight: float,
    stage_progress: float,
    stage_weight: float,
) -> float:
    return min(
        1.0,
        completed_weight + stage_weight * max(0.0, min(1.0, stage_progress)),
    )


class PipelineManager:
    """Orchestrates Phases 1 through 6 in a resilient, resumable production pipeline."""

    def __init__(self, project: Project):
        self.project = project

    def run_stage(
        self,
        name: str,
        function,
        *args,
        control: Optional[JobControl] = None,
        completed_weight: float = 0.0,
        **kwargs,
    ):
        if control is not None:
            control.check()
            control.update(
                stage=name,
                progress=completed_weight,
                message=f"Starting stage: {name}...",
            )

        if self.project.stage_complete(name):
            print(f"[SKIP] {name} already complete")
            if control is not None:
                control.update(
                    stage=name,
                    progress=completed_weight + STAGE_WEIGHTS.get(name, 0.0),
                    message=f"Stage {name} cached.",
                )
            return self.project.load()["stages"][name]["output"]

        self.project.update_stage(name, "running")
        try:
            print(f"[START] {name}")
            result = function(*args, control=control, **kwargs)

            if control is not None:
                control.check()

            self.project.update_stage(name, "complete", output=result)
            print(f"[DONE] {name}")

            if control is not None:
                control.update(
                    stage=name,
                    progress=completed_weight + STAGE_WEIGHTS.get(name, 0.0),
                    message=f"Completed stage: {name}",
                )

            return result
        except JobCancelledError:
            self.project.update_stage(name, "cancelled", error="Cancelled by user.")
            raise
        except Exception as exc:
            self.project.update_stage(name, "failed", error=str(exc))
            traceback.print_exc()
            raise

    def run(
        self,
        input_video: str | Path,
        style: str = "anime_creator",
        job: Optional[object] = None,
    ) -> Dict[str, Any]:
        project_root = self.project.root
        source = Path(input_video).resolve()

        if not source.exists():
            raise FileNotFoundError(f"Source video not found: {source}")

        control = JobControl(job)
        control.check()

        with ProjectLock(project_root):
            snapshot = ResourceMonitor.snapshot(str(project_root))
            healthy, errors = ResourceMonitor.healthy(snapshot)
            if not healthy:
                err_msg = "System is not ready:\n" + "\n".join(errors)
                logger.warning(err_msg)

            control.check()
            completed_weight = 0.0

            # PHASE 1: Video Input & Processing
            phase1_output = self.run_stage(
                "input",
                self._phase1,
                source,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["input"]

            # PHASE 2: Vision & Scene Understanding
            vision_output = self.run_stage(
                "vision",
                self._phase2,
                phase1_output,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["vision"]

            # PHASE 4: Character Consistency / Temporal Planning
            # Executed BEFORE artistic rendering to provide keyframe decisions & scene-cut boundaries
            consistency_output = self.run_stage(
                "consistency",
                self._phase4,
                phase1_output,
                vision_output,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["consistency"]

            # PHASE 5A: Lip-Sync Analysis & Viseme Timeline
            lipsync_output = self.run_stage(
                "lipsync",
                self._phase5_lipsync,
                phase1_output,
                vision_output,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["lipsync"]

            # PHASE 3: Artistic Style Engine
            # Receives Vision data, Consistency Temporal Plan, and Lip-Sync Timeline
            artistic_output = self.run_stage(
                "artistic",
                self._phase3,
                phase1_output,
                vision_output,
                consistency_output,
                style,
                project_root,
                lipsync_output,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["artistic"]

            # PHASE 5B: Media Composition & Multiplexing
            final_output = self.run_stage(
                "composition",
                self._phase5_compose,
                artistic_output,
                phase1_output,
                lipsync_output,
                consistency_output,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["composition"]

            # VALIDATION
            validation_output = self.run_stage(
                "validation",
                self._validate,
                final_output,
                project_root,
                control=control,
                completed_weight=completed_weight,
            )
            completed_weight += STAGE_WEIGHTS["validation"]

            control.update(
                stage="complete",
                progress=1.0,
                message="Video production complete!",
            )

            return {
                "final_video": str(final_output),
                "validation": str(validation_output),
            }

    # ------------------------------------
    # Phase adapters
    # ------------------------------------

    def _phase1(self, source: Path, root: Path, control: Optional[JobControl] = None) -> Path:
        output = root / "source" / source.name
        if source != output:
            shutil.copy2(str(source), str(output))
        return output

    def _phase2(self, source: Path, root: Path, control: Optional[JobControl] = None) -> Path:
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
        consistency: Path,
        style: str,
        root: Path,
        lipsync: Optional[Path] = None,
        control: Optional[JobControl] = None,
    ) -> Path:
        from src.art.keyframe_video_renderer import KeyframeVideoRenderer
        from src.art.types import StyleConfig

        output = root / "artistic" / "animated.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        config = StyleConfig(name=str(style))
        renderer = KeyframeVideoRenderer(config=config)
        renderer.render_video(
            input_path=str(source),
            vision_jsonl=str(vision),
            temporal_plan=str(consistency),
            output_path=str(output),
            quality_report_path=root / "consistency" / "identity_quality.json",
            control=control,
        )
        return output

    def _phase4(
        self,
        source: Path,
        vision: Path,
        root: Path,
        control: Optional[JobControl] = None,
    ) -> Path:
        from src.consistency.planner import TemporalPlanner

        output = root / "consistency" / "temporal_plan.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)

        planner = TemporalPlanner(keyframe_interval=12)
        planner.generate_plan(
            video_path=str(source),
            vision_jsonl_path=str(vision),
            output_plan_path=str(output),
        )
        return output

    def _phase5_lipsync(
        self,
        source: Path,
        vision: Path,
        root: Path,
        control: Optional[JobControl] = None,
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
        control: Optional[JobControl] = None,
    ) -> Path:
        from src.media.compose import compose_final_video

        output = root / "output" / "youtube_master.mp4"
        output.parent.mkdir(parents=True, exist_ok=True)

        partial_output = output.with_name(f"{output.stem}.partial{output.suffix}")

        compose_final_video(
            animated_video=str(artistic),
            audio_source=str(source),
            output=str(partial_output),
        )

        if partial_output.exists():
            partial_output.replace(output)

        return output

    def _validate(
        self,
        video: Path,
        root: Path,
        control: Optional[JobControl] = None,
    ) -> Path:
        from src.media.validate import validate_video

        result = validate_video(str(video))
        report = root / "reports" / "validation.json"
        report.parent.mkdir(parents=True, exist_ok=True)

        with open(str(report), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        if not result.get("valid", True):
            raise RuntimeError(f"Final video validation failed: {result.get('errors')}")

        return report
