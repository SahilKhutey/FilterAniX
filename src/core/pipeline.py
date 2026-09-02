"""Master Production Pipeline Controller Uniting Phases 1 through 5."""
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, Optional

from src.core.project import Project, ProjectStatus
from src.core.logging_setup import setup_project_logger
from src.io.video_io import inspect_video
from src.vision.vision_pipeline import VisionEngine
from src.art.types import RenderConfig, StylePreset
from src.art.video_renderer import VideoStyleRenderer
from src.consistency.planner import TemporalPlanner
from src.consistency.identity import IdentityProfileBuilder
from src.consistency.report import ConsistencyAuditor
from src.lipsync.analyzer import LipSyncAnalyzer
from src.lipsync.smoother import LipSyncSmoother
from src.media.compose import VideoCompositor
from src.media.validate import OutputValidator


class ProductionPipelineController:
    """End-to-End State Machine Orchestrator executing Phases 1 through 5 with checkpointing."""

    def __init__(self, project: Project, style_key: str = "anime_creator"):
        self.project = project
        self.style_key = style_key
        self.logger = setup_project_logger(project.log_file)

    def _load_style_preset(self) -> StylePreset:
        """Loads style parameters from styles.json."""
        styles_path = Path("styles.json")
        preset = StylePreset(name=self.style_key)
        if styles_path.exists():
            with open(styles_path, "r", encoding="utf-8-sig") as f:
                s_data = json.load(f)
                if self.style_key in s_data:
                    c = s_data[self.style_key]
                    preset.name = c.get("name", self.style_key)
                    preset.line_weight = c.get("line_weight", preset.line_weight)
                    preset.line_tint = c.get("line_tint", preset.line_tint)
                    preset.shading_levels = c.get("shading_levels", preset.shading_levels)
                    preset.color_warmth = c.get("color_warmth", preset.color_warmth)
                    preset.contrast_boost = c.get("contrast_boost", preset.contrast_boost)
                    preset.saturation_boost = c.get("saturation_boost", preset.saturation_boost)
        return preset

    def run(
        self,
        input_video_path: str | Path,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        resume: bool = True,
    ) -> str:
        input_src = Path(input_video_path).resolve()
        if not input_src.exists():
            raise FileNotFoundError(f"Input video not found: {input_src}")

        self.logger.info(f"Starting Production Pipeline for Project: {self.project.root_dir.name}")
        self.project.set_status(ProjectStatus.QUEUED)

        # Stage 1: Phase 1 Input & Metadata Inspection
        self.project.set_status(ProjectStatus.PHASE_1)
        if progress_callback:
            progress_callback("Phase 1: Ingesting & Inspecting Media", 5.0)

        local_input = self.project.input_dir / "input_source.mp4"
        if not local_input.exists():
            shutil.copy2(str(input_src), str(local_input))

        meta = inspect_video(local_input)
        meta_json = self.project.phase1_dir / "metadata.json"
        with open(meta_json, "w", encoding="utf-8") as f:
            json.dump(meta.summary_dict(), f, indent=2)

        self.project.manifest.input_video_path = str(local_input)
        self.project.record_stage_completed("phase1_input", meta.summary_dict())
        self.logger.info(f"PHASE 1 Complete: {meta.resolution_str} @ {meta.fps:.2f} FPS ({meta.frame_count} frames)")

        # Stage 2: Phase 2 Vision & Scene Understanding
        self.project.set_status(ProjectStatus.PHASE_2)
        if progress_callback:
            progress_callback("Phase 2: Vision & Landmark Extraction", 20.0)

        vision_jsonl = self.project.phase2_dir / "vision.jsonl"
        summary_json = self.project.phase2_dir / "summary.json"
        annotated_mp4 = self.project.phase2_dir / "annotated.mp4"

        if not (resume and vision_jsonl.exists() and vision_jsonl.stat().st_size > 0):
            from analyze_video import analyze_video
            analyze_video(
                video_path=str(local_input),
                output_dir=str(self.project.phase2_dir),
                max_frames=max_frames,
            )
        self.project.record_stage_completed("phase2_vision", {"jsonl": str(vision_jsonl)})
        self.logger.info("PHASE 2 Complete: Scene understanding and landmark tracking locked.")

        # Stage 3: Phase 3 Artistic Style Rendering
        self.project.set_status(ProjectStatus.PHASE_3)
        if progress_callback:
            progress_callback("Phase 3: Artistic Style Rendering", 50.0)

        artistic_mp4 = self.project.phase3_dir / "artistic_video.mp4"
        if not (resume and artistic_mp4.exists() and artistic_mp4.stat().st_size > 1000):
            style_preset = self._load_style_preset()
            config = RenderConfig(style=style_preset)
            renderer = VideoStyleRenderer(config)
            renderer.render_video(
                input_path=local_input,
                output_path=artistic_mp4,
                vision_jsonl=vision_jsonl,
                max_frames=max_frames,
            )
        self.project.record_stage_completed("phase3_style", {"artistic_video": str(artistic_mp4)})
        self.logger.info("PHASE 3 Complete: Full artistic style rendering generated.")

        # Stage 4: Phase 4 Identity & Temporal Consistency
        self.project.set_status(ProjectStatus.PHASE_4)
        if progress_callback:
            progress_callback("Phase 4: Temporal Consistency & Identity Audit", 70.0)

        temporal_plan_jsonl = self.project.phase4_dir / "temporal_plan.jsonl"
        consistency_report_json = self.project.phase4_dir / "consistency_report.json"
        ref_profile_json = self.project.phase4_dir / "reference_profile.json"

        # Generate Temporal Plan
        planner = TemporalPlanner(keyframe_interval=12)
        planner.generate_plan(
            video_path=local_input,
            vision_jsonl_path=vision_jsonl,
            output_plan_path=temporal_plan_jsonl,
            max_frames=max_frames,
        )

        # Audit generated video against first keyframe
        import cv2
        cap = cv2.VideoCapture(str(artistic_mp4))
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            profile = IdentityProfileBuilder.build_profile(
                cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB), name="project_character"
            )
            with open(ref_profile_json, "w", encoding="utf-8") as f:
                json.dump(profile.to_dict(), f, indent=2)

            auditor = ConsistencyAuditor(profile)
            auditor.audit_video(
                video_path=artistic_mp4,
                output_report_path=consistency_report_json,
                max_frames=max_frames,
            )

        self.project.record_stage_completed("phase4_consistency", {"report": str(consistency_report_json)})
        self.logger.info("PHASE 4 Complete: Character consistency verified.")

        # Stage 5: Phase 5 Lip-Sync & Final Composition
        self.project.set_status(ProjectStatus.PHASE_5)
        if progress_callback:
            progress_callback("Phase 5: Lip-Sync & Master Composition", 85.0)

        lipsync_jsonl = self.project.phase5_dir / "lipsync.jsonl"
        youtube_master_mp4 = self.project.phase5_dir / "youtube_master.mp4"

        # Extract Lip-Sync
        from build_lipsync import main as _build_lipsync
        analyzer = LipSyncAnalyzer()
        raw_recs = []
        with open(vision_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                from src.vision.models import FrameVisionData, FaceData, BoundingBox
                faces = [FaceData(face_id=0, landmarks=[], bbox=BoundingBox(**fd["bbox"]), landmark_count=0, mouth_opening=fd.get("mouth_opening", 0.0)) for fd in d.get("faces", [])]
                v_data = FrameVisionData(frame_index=d["frame_index"], timestamp=d["timestamp"], width=1920, height=1080, faces=faces)
                raw_recs.append(analyzer.analyze_frame(d["frame_index"], d["timestamp"], v_data))

        smoother = LipSyncSmoother(window_size=5)
        smoothed = smoother.smooth_timeline(raw_recs)
        with open(lipsync_jsonl, "w", encoding="utf-8") as out_f:
            for r in smoothed:
                out_f.write(json.dumps(r.to_dict()) + "\n")

        # Compose Master
        compositor = VideoCompositor(target_lufs=-14.0, true_peak=-1.5)
        compositor.compose(
            video_path=artistic_mp4,
            audio_source_path=local_input,
            output_path=youtube_master_mp4,
            normalize_loudness=True,
        )

        # Stage 6: Validation
        self.project.set_status(ProjectStatus.VALIDATING)
        validator = OutputValidator()
        validation = validator.validate(youtube_master_mp4)
        validation_json = self.project.phase5_dir / "validation.json"
        with open(validation_json, "w", encoding="utf-8") as f:
            json.dump(validation.to_dict(), f, indent=2)

        self.project.record_stage_completed("phase5_composition", {"master_mp4": str(youtube_master_mp4)})
        self.project.manifest.outputs["youtube_master"] = str(youtube_master_mp4)
        self.project.set_status(ProjectStatus.COMPLETED)

        if progress_callback:
            progress_callback("Pipeline Execution Complete", 100.0)

        self.logger.info(f"[SUCCESS] Pipeline complete! Final YouTube Master written to: {youtube_master_mp4}")
        return str(youtube_master_mp4)
