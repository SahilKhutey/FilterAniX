from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.art.diffusion_renderer import DiffusionRenderer
from src.art.frame_propagator import FramePropagator
from src.art.keyframe_cache import KeyframeCache
from src.art.keyframe_scheduler import KeyframeScheduler
from src.art.math_engine import MathematicalStyleEngine
from src.art.render_metrics import RenderMetrics
from src.art.render_schedule import load_temporal_plan
from src.art.style_controller import StyleController
from src.art.types import StyleConfig, RendererBackend
from src.consistency.identity_engine import IdentityEngine
from src.consistency.quality_report import IdentityQualityReport
from src.core.cancellation import JobControl


class KeyframeVideoRenderer:
    """
    Deterministic Mathematical Video Renderer with Keyframe Telemetry and Quality Auditing.
    Transforms every pixel of every frame mathematically, with temporal stabilization and scene-cut isolation.
    """

    def __init__(
        self,
        config: StyleConfig | None = None,
    ):
        self.config = (
            config
            or StyleConfig()
        )

        self.style_controller = (
            StyleController(self.config)
        )

        self.math_engine = (
            MathematicalStyleEngine(self.config)
        )

        self.diffusion_renderer = (
            DiffusionRenderer(self.config)
        )

        self.scheduler = KeyframeScheduler(
            minimum_interval=self.config.keyframe_interval,
            motion_threshold=0.22,
        )

        self.propagator = FramePropagator(
            blend=self.config.temporal_blend
        )

        self.identity_engine = IdentityEngine(
            warning_threshold=self.config.identity_warning_threshold,
            severe_threshold=self.config.identity_severe_threshold,
            max_retries=self.config.identity_max_retries,
            bank_size=self.config.identity_reference_bank_size,
        )

    def render_video(
        self,
        input_path: str | Path,
        vision_jsonl: str | Path,
        temporal_plan: str | Path,
        output_path: str | Path,
        cache_dir: str | Path | None = None,
        reference_rgb: np.ndarray | None = None,
        quality_report_path: str | Path | None = None,
        job: Optional[object] = None,
        control: Optional[JobControl] = None,
    ) -> dict[str, Any]:

        if control is None:
            control = JobControl(job)

        input_path = Path(input_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        partial_output_path = output_path.with_name(
            f"{output_path.stem}.partial{output_path.suffix}"
        )

        if cache_dir is None:
            cache_dir = (
                output_path.parent
                / "keyframes"
            )

        cache = KeyframeCache(cache_dir)

        plan_map = load_temporal_plan(
            temporal_plan
        )

        ordered_plan = [
            plan_map[index]
            for index in sorted(plan_map)
        ]

        decisions = self.scheduler.build(
            ordered_plan
        )

        vision_map = self._load_vision(
            vision_jsonl
        )

        capture = cv2.VideoCapture(
            str(input_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {input_path}"
            )

        fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 30.0

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        writer = cv2.VideoWriter(
            str(partial_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()

            raise RuntimeError(
                f"Unable to create output: {partial_output_path}"
            )

        total_frames = len(decisions)
        metrics = RenderMetrics()
        quality_report = IdentityQualityReport(total_frames=total_frames)

        previous_scene: int | None = None
        self.math_engine.reset_temporal()

        start_time = time.time()
        success = False

        try:

            for idx, decision in enumerate(tqdm(
                decisions,
                desc="Mathematical Style Engine Processing",
                unit="frame",
            )):

                control.check()

                frame_index = (
                    decision.frame_index
                )

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_index,
                )

                ok, bgr = capture.read()

                if not ok:
                    continue

                source_rgb = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB,
                )

                metrics.total_frames += 1

                vision = vision_map.get(
                    frame_index,
                    {},
                )

                # Extract face data for identity extraction
                face_bbox = None
                landmarks = None
                if vision and "faces" in vision and vision["faces"]:
                    f0 = vision["faces"][0]
                    face_bbox = f0.get("bbox")
                    landmarks = f0.get("landmarks")

                scene_changed = (
                    previous_scene is not None
                    and decision.scene_id != previous_scene
                )

                scene_reset = (
                    decision.is_scene_cut
                    or scene_changed
                )

                if scene_reset:
                    self.math_engine.reset_temporal()
                    metrics.scene_cuts += 1

                # Select best scene-specific reference from Identity Bank
                ref_candidate = self.identity_engine.reference_for(
                    decision.scene_id,
                    frame_index,
                )
                effective_reference_rgb = (
                    ref_candidate
                    if ref_candidate is not None
                    else reference_rgb
                )

                # Check if explicit diffusion backend requested
                use_diffusion = (
                    self.config.backend == RendererBackend.DIFFUSERS
                    and self.diffusion_renderer.pipeline is not None
                    and decision.is_keyframe
                )

                if use_diffusion:
                    control_map = (
                        self.style_controller.build_control_map(
                            source_rgb,
                            vision,
                        )
                    )
                    art_rgb = self.diffusion_renderer.render(
                        rgb=source_rgb,
                        control_map=control_map,
                        vision_data=None,
                        reference_rgb=effective_reference_rgb,
                        reference_strength=decision.reference_strength,
                        denoise_strength=self.config.denoise_strength,
                    )
                    metrics.diffusion_frames += 1
                else:
                    # Primary Mathematical Style Engine (Every Pixel Transformed Deterministically)
                    art_rgb = self.math_engine.render(
                        rgb=source_rgb,
                        vision_data=vision,
                        scene_cut=scene_reset,
                        stabilize=True,
                        reference_rgb=effective_reference_rgb,
                    )
                    metrics.fallback_frames += 1

                if decision.is_keyframe:
                    metrics.keyframes += 1
                    cache.save(frame_index, art_rgb)
                else:
                    metrics.propagated_frames += 1

                # Identity Evaluation & Telemetry
                if self.config.identity_enabled and (frame_index % self.config.identity_evaluation_interval == 0):
                    eval_res = self.identity_engine.evaluate(
                        frame_index=frame_index,
                        scene_id=decision.scene_id,
                        image_rgb=art_rgb,
                        face_bbox=face_bbox,
                        landmarks=landmarks,
                    )
                    quality_report.add(
                        score=eval_res.metric.overall,
                        warning=eval_res.metric.warning,
                        severe=eval_res.metric.severe_drift,
                    )

                writer.write(
                    cv2.cvtColor(
                        art_rgb,
                        cv2.COLOR_RGB2BGR,
                    )
                )

                previous_scene = decision.scene_id

                # Periodic Progress and ETA reporting
                elapsed = time.time() - start_time
                frames_done = idx + 1
                fps_val = frames_done / max(elapsed, 0.001)
                remaining_frames = total_frames - frames_done
                eta_sec = remaining_frames / fps_val if fps_val > 0 else 0.0
                stg_progress = frames_done / max(total_frames, 1)

                control.update(
                    stage="artistic",
                    progress=stg_progress,
                    current_frame=frames_done,
                    total_frames=total_frames,
                    fps=fps_val,
                    eta_seconds=eta_sec,
                    elapsed_seconds=elapsed,
                    message=f"Mathematical Render: {frames_done}/{total_frames} frames",
                )

            success = True

        finally:

            capture.release()
            writer.release()

            if success and partial_output_path.exists():
                partial_output_path.replace(output_path)
            elif not success and partial_output_path.exists():
                partial_output_path.unlink(missing_ok=True)

        metrics.render_seconds = (
            time.time() - start_time
        )

        metrics.save(
            output_path.parent
            / "render_metrics.json"
        )

        report_target = (
            Path(quality_report_path)
            if quality_report_path is not None
            else output_path.parent.parent / "consistency" / "identity_quality.json"
        )
        quality_report.save(report_target)

        return {
            "output": str(output_path),
            "frames": metrics.total_frames,
            "keyframes": metrics.keyframes,
            "propagated_frames": (
                metrics.propagated_frames
            ),
            "diffusion_frames": (
                metrics.diffusion_frames
            ),
            "fallback_frames": (
                metrics.fallback_frames
            ),
            "scene_cuts": metrics.scene_cuts,
            "render_seconds": (
                metrics.render_seconds
            ),
            "keyframe_ratio": (
                metrics.keyframe_ratio
            ),
            "propagation_ratio": (
                metrics.propagation_ratio
            ),
            "identity_quality": str(report_target),
        }

    @staticmethod
    def _load_vision(
        path: str | Path,
    ) -> dict[int, dict[str, Any]]:

        result = {}

        path = Path(path)

        if not path.exists():
            return result

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            for line in handle:

                line = line.strip()

                if not line:
                    continue

                item = __import__(
                    "json"
                ).loads(line)

                index = int(
                    item.get(
                        "frame_index",
                        len(result),
                    )
                )

                result[index] = item

        return result
