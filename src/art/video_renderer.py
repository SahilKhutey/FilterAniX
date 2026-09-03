from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np
from tqdm import tqdm

from .opencv_renderer import OpenCVArtRenderer
from .style_controller import StyleController
from .temporal import TemporalStabilizer
from .types import RenderConfig, StyleConfig
from src.art.style_engine import StyleEngine
from src.core.models import ProcessingProgress
from src.io.video_io import inspect_video, create_video_writer, merge_audio_and_video
from src.vision.models import (
    FrameVisionData,
    FaceData,
    PoseData,
    HandData,
    Landmark,
    BoundingBox,
    PersonMaskData,
    MotionData,
)


from src.consistency.types import RenderDecision
from src.lipsync.analyzer import LipSyncRecord


from src.art.render_context import RenderContext
from src.art.mouth_renderer import MouthRenderer
from src.art.keyframe_scheduler import KeyframeScheduler, KeyframeDecision
from src.art.keyframe_cache import KeyframeCache
from src.art.keyframe_renderer import KeyframeRenderer
from src.art.frame_propagator import FramePropagator
from src.art.render_schedule import load_temporal_plan
from src.art.render_metrics import RenderMetrics


def get_mouth_bbox(
    vision_data: Optional[FrameVisionData],
    width: int,
    height: int,
) -> Optional[tuple[int, int, int, int]]:
    if vision_data is None or not vision_data.faces:
        return None

    face = vision_data.faces[0]
    if not face.landmarks:
        bx = int(face.bbox.x * width)
        by = int(face.bbox.y * height)
        bw = int(face.bbox.width * width)
        bh = int(face.bbox.height * height)
        return (bx + bw // 4, by + int(bh * 0.65), bx + int(bw * 0.75), by + int(bh * 0.95))

    if len(face.landmarks) >= 468:
        lip_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
        ]
        lip_lms = [face.landmarks[i] for i in lip_indices if i < len(face.landmarks)]
        if lip_lms:
            lxs = [p.x * width for p in lip_lms]
            lys = [p.y * height for p in lip_lms]
            pad_x = 4
            pad_y = 4
            return (
                max(0, int(min(lxs) - pad_x)),
                max(0, int(min(lys) - pad_y)),
                min(width, int(max(lxs) + pad_x)),
                min(height, int(max(lys) + pad_y)),
            )

    xs = [lm.x * width for lm in face.landmarks]
    ys = [lm.y * height for lm in face.landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    fh = max_y - min_y
    return (
        int(min_x + (max_x - min_x) * 0.25),
        int(min_y + fh * 0.65),
        int(min_x + (max_x - min_x) * 0.75),
        int(min_y + fh * 0.95),
    )


class VideoRenderer:
    """Canonical video-level artistic renderer coordinating KeyframeScheduler, Diffusion, Propagation, and Temporal Stabilization."""

    def __init__(
        self,
        style_config: StyleConfig | None = None,
        config: StyleConfig | None = None,
    ):
        self.style_config = config or style_config or StyleConfig()
        self.style_controller = StyleController(self.style_config)
        self.fast_renderer = OpenCVArtRenderer()
        self.temporal = TemporalStabilizer(self.style_config.temporal_blend)
        self.mouth_renderer = MouthRenderer()

    def render_video(
        self,
        video_path: str | Path,
        vision_jsonl_path: str | Path,
        output_path: str | Path,
        temporal_plan_path: Optional[str | Path] = None,
        lipsync_jsonl_path: Optional[str | Path] = None,
        temporal_plan_jsonl: Optional[str | Path] = None,
        lipsync_jsonl: Optional[str | Path] = None,
    ) -> dict[str, Any]:
        return self.render(
            input_video=video_path,
            vision_jsonl=vision_jsonl_path,
            output_video=output_path,
            temporal_plan_path=temporal_plan_jsonl or temporal_plan_path,
            lipsync_jsonl_path=lipsync_jsonl or lipsync_jsonl_path,
        )

    def render(
        self,
        input_video: str | Path,
        vision_jsonl: str | Path,
        output_video: str | Path,
        temporal_plan_path: Optional[str | Path] = None,
        lipsync_jsonl_path: Optional[str | Path] = None,
        temporal_plan_jsonl: Optional[str | Path] = None,
        lipsync_jsonl: Optional[str | Path] = None,
    ) -> dict[str, Any]:

        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)

        plan_p = temporal_plan_jsonl or temporal_plan_path
        lip_p = lipsync_jsonl or lipsync_jsonl_path

        vision_frames = self._load_vision(str(vision_jsonl))
        temporal_map = self._load_jsonl_map(plan_p)
        lipsync_records = self._load_lipsync(lip_p)

        # 1. Initialize P1 Scheduler, Cache, Keyframe Renderer, and Propagator
        scheduler = KeyframeScheduler(
            minimum_interval=self.style_config.keyframe_interval,
        )
        schedule_items = scheduler.build(list(temporal_map.values()))
        schedule_map = {item.frame_index: item for item in schedule_items}

        keyframe_cache = KeyframeCache(output_video.parent / "keyframes")
        keyframe_renderer = KeyframeRenderer(self.style_config)
        propagator = FramePropagator(blend=self.style_config.temporal_blend)
        metrics = RenderMetrics()

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {input_video}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        self.temporal.reset()
        frame_index = 0
        start_time = time.time()

        previous_source: Optional[np.ndarray] = None
        previous_art: Optional[np.ndarray] = None
        previous_scene_id: Optional[int] = None
        reference_rgb: Optional[np.ndarray] = None

        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                vision = (
                    vision_frames[frame_index]
                    if frame_index < len(vision_frames)
                    else {}
                )

                decision = schedule_map.get(frame_index)
                if decision is None:
                    decision = KeyframeDecision(
                        frame_index=frame_index,
                        scene_id=0,
                        is_keyframe=(frame_index == 0 or frame_index % self.style_config.keyframe_interval == 0),
                        is_scene_cut=(frame_index == 0),
                        motion_score=0.0,
                        reference_strength=0.55,
                        reason="default",
                    )

                lip_rec = lipsync_records.get(frame_index)
                viseme = lip_rec.viseme if lip_rec else "closed"

                # Scene cut or scene boundary isolation
                if decision.is_scene_cut or (previous_scene_id is not None and decision.scene_id != previous_scene_id):
                    self.temporal.reset()
                    previous_source = None
                    previous_art = None
                    metrics.scene_cuts += 1

                # Construct FrameVisionData object for face/pose/mouth awareness
                vision_obj = None
                if vision and "faces" in vision:
                    faces = []
                    for f_d in vision.get("faces", []):
                        b_raw = f_d.get("bbox") or {}
                        bbox = BoundingBox(
                            x=float(b_raw.get("x", 0.0)),
                            y=float(b_raw.get("y", 0.0)),
                            width=float(b_raw.get("width", 0.0)),
                            height=float(b_raw.get("height", 0.0)),
                        )
                        lms = [
                            Landmark(
                                x=float(lm.get("x", 0.0)),
                                y=float(lm.get("y", 0.0)),
                                z=float(lm.get("z", 0.0)),
                                visibility=float(lm.get("visibility", 1.0)),
                            )
                            for lm in f_d.get("landmarks", [])
                        ]
                        faces.append(
                            FaceData(
                                face_id=int(f_d.get("face_id", 0)),
                                landmarks=lms,
                                bbox=bbox,
                                landmark_count=f_d.get("landmark_count", len(lms)),
                                mouth_opening=float(f_d.get("mouth_opening", f_d.get("mouth_open", 0.0))),
                            )
                        )
                    vision_obj = FrameVisionData(
                        frame_index=frame_index,
                        timestamp=frame_index / fps,
                        width=width,
                        height=height,
                        faces=faces,
                    )

                # Build control map for ControlNet / structural guidance
                control_map = self.style_controller.build_control_map(frame_bgr, vision)

                # Render: Keyframe vs Optical Propagation
                if decision.is_keyframe:
                    metrics.keyframes += 1
                    if keyframe_cache.exists(frame_index):
                        animated = keyframe_cache.load(frame_index)
                        metrics.fallback_frames += 1
                    else:
                        kf_res = keyframe_renderer.render(
                            frame_index=frame_index,
                            rgb=rgb,
                            control_map=control_map,
                            vision_data=vision_obj,
                            reference_rgb=reference_rgb,
                            reference_strength=decision.reference_strength,
                        )
                        animated = kf_res.frame
                        keyframe_cache.save(frame_index, animated)
                        if kf_res.backend == "diffusion" and not kf_res.used_fallback:
                            metrics.diffusion_frames += 1
                        else:
                            metrics.fallback_frames += 1

                    if reference_rgb is None:
                        reference_rgb = rgb.copy()
                else:
                    # Intermediate frame: source-guided optical flow warping
                    if previous_source is not None and previous_art is not None:
                        animated = propagator.warp(
                            previous_source=previous_source,
                            current_source=rgb,
                            previous_art=previous_art,
                        )
                        metrics.propagated_frames += 1
                    else:
                        # Fallback to fast renderer
                        animated = self.fast_renderer.render(rgb, vision_data=vision_obj, lipsync_record=lip_rec)
                        metrics.fallback_frames += 1

                # Apply procedural mouth renderer based on Lip-Sync timeline
                mouth_bbox = get_mouth_bbox(vision_obj, width, height)
                if mouth_bbox is not None and viseme != "closed":
                    animated = self.mouth_renderer.render(
                        frame=animated,
                        mouth_bbox=mouth_bbox,
                        viseme=viseme,
                    )

                # Temporal stabilization
                animated = self.temporal.apply(
                    animated,
                    scene_id=decision.scene_id,
                    scene_cut=decision.is_scene_cut,
                )

                # Prepare state for next frame
                previous_source = rgb.copy()
                previous_art = animated.copy()
                previous_scene_id = decision.scene_id

                # Write RGB frame as BGR to output video
                writer.write(cv2.cvtColor(animated, cv2.COLOR_RGB2BGR))
                frame_index += 1
                metrics.total_frames = frame_index
        finally:
            capture.release()
            writer.release()

        metrics.render_seconds = round(time.time() - start_time, 2)
        metrics.save(output_video.parent / "render_metrics.json")

        return {
            "frames": frame_index,
            "fps": fps,
            "width": width,
            "height": height,
            "output": str(output_video),
            "metrics": str(output_video.parent / "render_metrics.json"),
        }

    @staticmethod
    def _load_jsonl_map(
        path: str | Path | None,
    ) -> dict[int, dict[str, Any]]:
        if path is None:
            return {}

        path = Path(path)
        if not path.exists():
            return {}

        result: dict[int, dict[str, Any]] = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                frame_index = int(data.get("frame_index", data.get("frame", 0)))
                result[frame_index] = data
        return result

    @staticmethod
    def _load_vision(path: str) -> list[dict[str, Any]]:
        result = []
        p = Path(path) if path else None
        if not p or not p.exists():
            return result

        with open(str(p), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                result.append(json.loads(line))
        return result

    @staticmethod
    def _load_temporal_plan(path: Optional[str | Path]) -> Dict[int, RenderDecision]:
        decisions: Dict[int, RenderDecision] = {}
        if not path or not Path(path).exists():
            return decisions

        with open(str(path), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                idx = d.get("frame_index", 0)
                decisions[idx] = RenderDecision(
                    frame_index=idx,
                    timestamp=float(d.get("timestamp", 0.0)),
                    scene_id=int(d.get("scene_id", 0)),
                    is_scene_cut=bool(d.get("is_scene_cut", False)),
                    is_keyframe=bool(d.get("is_keyframe", False)),
                    motion_score=float(d.get("motion_score", 0.0)),
                    reference_strength=float(d.get("reference_strength", 1.0)),
                    preserve_previous=bool(d.get("preserve_previous", True)),
                    similarity_warning=bool(d.get("similarity_warning", False)),
                    reason=d.get("reason", "standard"),
                )
        return decisions

    @staticmethod
    def _load_lipsync(path: Optional[str | Path]) -> Dict[int, LipSyncRecord]:
        records: Dict[int, LipSyncRecord] = {}
        if not path or not Path(path).exists():
            return records

        with open(str(path), "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                idx = d.get("frame_index", d.get("frame", 0))
                records[idx] = LipSyncRecord(
                    frame_index=idx,
                    timestamp=float(d.get("timestamp", 0.0)),
                    mouth_open_ratio=float(d.get("mouth_open_ratio", d.get("mouth_open", 0.0))),
                    viseme=d.get("viseme", d.get("state", "closed")),
                )
        return records


class VideoStyleRenderer:
    """Full-pipeline artistic style video renderer with FFmpeg audio sync."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.style_engine = StyleEngine(self.config)

    def _load_vision_jsonl(self, jsonl_path: Optional[str | Path]) -> Dict[int, FrameVisionData]:
        if not jsonl_path or not Path(jsonl_path).exists():
            return {}

        vision_map: Dict[int, FrameVisionData] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                idx = d.get("frame_index", 0)

                faces = []
                for f_d in d.get("faces", []):
                    bbox = BoundingBox(**f_d["bbox"]) if f_d.get("bbox") else BoundingBox(0, 0, 0, 0)
                    lms = [Landmark(**lm) for lm in f_d.get("landmarks", [])]
                    faces.append(
                        FaceData(
                            face_id=f_d.get("face_id", 0),
                            landmarks=lms,
                            bbox=bbox,
                            landmark_count=f_d.get("landmark_count", len(lms)),
                            mouth_opening=f_d.get("mouth_opening", 0.0),
                            left_eye_ear=f_d.get("left_eye_ear", 0.0),
                            right_eye_ear=f_d.get("right_eye_ear", 0.0),
                        )
                    )

                pose = None
                if d.get("pose"):
                    p_d = d["pose"]
                    bbox = BoundingBox(**p_d["bbox"]) if p_d.get("bbox") else BoundingBox(0, 0, 0, 0)
                    lms = [Landmark(**lm) for lm in p_d.get("landmarks", [])]
                    pose = PoseData(
                        landmarks=lms,
                        bbox=bbox,
                        landmark_count=p_d.get("landmark_count", len(lms)),
                        torso_center=p_d.get("torso_center"),
                    )

                hands = []
                for h_d in d.get("hands", []):
                    bbox = BoundingBox(**h_d["bbox"]) if h_d.get("bbox") else BoundingBox(0, 0, 0, 0)
                    lms = [Landmark(**lm) for lm in h_d.get("landmarks", [])]
                    hands.append(
                        HandData(
                            label=h_d.get("label", "Unknown"),
                            confidence=h_d.get("confidence", 0.9),
                            landmarks=lms,
                            bbox=bbox,
                        )
                    )

                person_mask = None
                if d.get("person_mask"):
                    m_d = d["person_mask"]
                    bbox = BoundingBox(**m_d["bbox"]) if m_d.get("bbox") else None
                    person_mask = PersonMaskData(
                        threshold=m_d.get("threshold", 0.5),
                        coverage=m_d.get("coverage", 0.0),
                        bbox=bbox,
                    )

                motion = MotionData(**d.get("motion", {})) if "motion" in d else MotionData()

                frame_vision = FrameVisionData(
                    frame_index=idx,
                    timestamp=d.get("timestamp", 0.0),
                    width=d.get("width", 1920),
                    height=d.get("height", 1080),
                    faces=faces,
                    pose=pose,
                    hands=hands,
                    person_mask=person_mask,
                    motion=motion,
                    objects=[],
                )
                vision_map[idx] = frame_vision

        return vision_map

    def render_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        vision_jsonl: Optional[str | Path] = None,
        reference_image_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        side_by_side: bool = False,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
        temporal_plan_path: Optional[str | Path] = None,
        lipsync_jsonl: Optional[str | Path] = None,
    ) -> str:
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = inspect_video(input_path)
        total_frames = metadata.frame_count
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        vision_map = self._load_vision_jsonl(vision_jsonl)
        temporal_decisions = VideoRenderer._load_temporal_plan(temporal_plan_path)
        lipsync_records = VideoRenderer._load_lipsync(lipsync_jsonl)

        ref_rgb = None
        ref_path = reference_image_path or self.config.reference_image_path
        if ref_path and Path(ref_path).exists():
            ref_bgr = cv2.imread(str(ref_path))
            if ref_bgr is not None:
                ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

        temp_silent_video = output_path.parent / f"temp_art_silent_{int(time.time()*1000)}.mp4"
        out_w = metadata.width * 2 if side_by_side else metadata.width
        out_h = metadata.height

        writer = create_video_writer(
            output_path=temp_silent_video,
            width=out_w,
            height=out_h,
            fps=metadata.fps,
            fourcc_str="mp4v",
        )

        cap = cv2.VideoCapture(str(input_path))
        self.style_engine.reset_temporal()

        start_time = time.time()
        pbar = tqdm(total=total_frames, desc="Rendering Art Video", unit="frame")

        for frame_idx in range(total_frames):
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            vision_data = vision_map.get(frame_idx, None)
            decision = temporal_decisions.get(frame_idx, None)
            lipsync = lipsync_records.get(frame_idx, None)

            art_rgb = self.style_engine.render_frame(
                rgb=rgb,
                vision_data=vision_data,
                reference_rgb=ref_rgb,
                stabilize=True,
                lipsync_record=lipsync,
                temporal_decision=decision,
            )

            if side_by_side:
                combined = np.hstack([rgb, art_rgb])
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            else:
                writer.write(cv2.cvtColor(art_rgb, cv2.COLOR_RGB2BGR))

            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames - 1):
                now = time.time()
                elapsed = max(0.001, now - start_time)
                fps_val = (frame_idx + 1) / elapsed
                percent = ((frame_idx + 1) / total_frames) * 100.0
                eta = (total_frames - (frame_idx + 1)) / fps_val if fps_val > 0 else 0.0
                progress_callback(
                    ProcessingProgress(
                        current_frame=frame_idx + 1,
                        total_frames=total_frames,
                        percent=percent,
                        fps=fps_val,
                        elapsed_sec=elapsed,
                        eta_sec=eta,
                        status_message=f"Rendering frame {frame_idx+1}/{total_frames} ({percent:.1f}%)",
                    )
                )

            pbar.update(1)

        pbar.close()
        cap.release()
        writer.release()

        merge_audio_and_video(
            silent_video_path=temp_silent_video,
            audio_source_path=input_path,
            final_output_path=output_path,
            has_audio=metadata.has_audio,
        )

        if temp_silent_video.exists():
            temp_silent_video.unlink(missing_ok=True)

        return str(output_path)
