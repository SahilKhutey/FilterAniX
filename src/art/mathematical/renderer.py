from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .config import MathematicalAnimeStyle
from .color_field import MathematicalColorField, ColorFieldResult
from .tone_field import MathematicalToneField, ToneFieldResult
from .palette_field import MathematicalPaletteField, PaletteFieldResult
from .edge_field import MathematicalEdgeField, EdgeFieldResult
from .shadow_highlight_field import MathematicalShadowHighlightField, ShadowHighlightFieldResult
from .geometry_types import GeometryObservation, GeometryBox
from .geometry_field import MathematicalGeometryField, GeometryFieldResult
from .face_field import MathematicalFaceField, FaceFieldResult
from .lighting_field import MathematicalLightingField, LightingFieldResult
from .temporal_types import TemporalObservation
from .temporal_field import MathematicalTemporalField, TemporalFieldResult
from .vision_adapter import adapt_vision_frame
from src.core.errors import RenderingError


@dataclass
class MathematicalRenderResult:
    """Complete multi-field output of the Mathematical Style Engine."""

    output_rgb: np.ndarray

    mth02: Optional[ColorFieldResult] = None
    mth03: Optional[ToneFieldResult] = None
    mth04: Optional[PaletteFieldResult] = None
    mth05: Optional[EdgeFieldResult] = None
    mth06: Optional[ShadowHighlightFieldResult] = None
    mth07: Optional[GeometryFieldResult] = None
    mth08: Optional[FaceFieldResult] = None
    mth09: Optional[LightingFieldResult] = None
    mth10: Optional[TemporalFieldResult] = None


class MathematicalRenderer:
    """
    Mathematical Anime Engine Orchestrator.

    Executes the canonical Phase-3 mathematical sequence:
        MTH-02 Color Field
             ↓
        MTH-03 Tone / Luminance Field
             ↓
        MTH-04 Palette Field
             ↓
        MTH-05 Edge / Line Field
             ↓
        MTH-06 Shadow / Highlight Field
             ↓
        MTH-07 Character / Geometry Field
             ↓
        MTH-08 Face / Facial Feature Field
             ↓
        MTH-09 Cinematic Lighting Field
             ↓
        MTH-10 Temporal Field

    100% deterministic, CPU-accelerated, zero diffusion or AI keyframes.
    """

    def __init__(
        self,
        style: Optional[MathematicalAnimeStyle] = None,
        event_bus: Optional[Any] = None,
    ) -> None:
        self.style = style or MathematicalAnimeStyle.creator_anime()
        self.event_bus = event_bus

        # Initialize all 9 mathematical stage engines
        self.color_field = MathematicalColorField(self.style)
        self.tone_field = MathematicalToneField(self.style)
        self.palette_field = MathematicalPaletteField(self.style)
        self.edge_field = MathematicalEdgeField(self.style)
        self.shadow_field = MathematicalShadowHighlightField(self.style)
        self.geometry_field = MathematicalGeometryField(self.style)
        self.face_field = MathematicalFaceField(self.style)
        self.lighting_field = MathematicalLightingField(self.style)
        self.temporal_field = MathematicalTemporalField(self.style)

        self._previous_input_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets all temporal history and internal state."""
        self.temporal_field.reset()
        self._previous_input_gray = None

    def _prepare_geometry_observation(
        self,
        vision: Any,
        height: int,
        width: int,
    ) -> GeometryObservation:
        """Ensures a valid GeometryObservation matching current frame dimensions."""
        if isinstance(vision, GeometryObservation):
            if vision.width == width and vision.height == height:
                return vision
            # Resolution mismatch - recreate with target resolution
            return GeometryObservation(width=width, height=height)

        if vision is not None:
            try:
                obs = adapt_vision_frame(vision)
                if obs.width == width and obs.height == height:
                    return obs
            except Exception:
                pass

        # Default empty geometry observation
        return GeometryObservation(
            width=width,
            height=height,
            face_box=None,
            face_landmarks=[],
            pose_landmarks=[],
            hand_landmarks=[],
            person_mask=np.ones((height, width), dtype=np.float32),
        )

    def render(
        self,
        frame_rgb: np.ndarray,
        vision: Any = None,
        temporal_observation: Optional[TemporalObservation] = None,
    ) -> MathematicalRenderResult:
        """
        Executes the full deterministic MTH-02 → MTH-10 transformation on a single RGB frame.
        """
        if not isinstance(frame_rgb, np.ndarray):
            raise TypeError("frame_rgb must be a numpy.ndarray")

        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError(f"frame_rgb must have shape (H, W, 3), got {frame_rgb.shape}")

        height, width = frame_rgb.shape[:2]
        geom_obs = self._prepare_geometry_observation(vision, height, width)

        # --------------------------------------------------------------
        # MTH-02: Color Field
        # --------------------------------------------------------------
        r02 = self.color_field.transform(frame_rgb)
        current = r02.output_rgb

        # --------------------------------------------------------------
        # MTH-03: Tone / Luminance Field
        # --------------------------------------------------------------
        r03 = self.tone_field.transform(current)
        current = r03.output_rgb

        # --------------------------------------------------------------
        # MTH-04: Palette Field
        # --------------------------------------------------------------
        r04 = self.palette_field.transform(current)
        current = r04.output_rgb

        # --------------------------------------------------------------
        # MTH-05: Edge / Line Field
        # --------------------------------------------------------------
        r05 = self.edge_field.transform(current)
        current = r05.output_rgb

        # --------------------------------------------------------------
        # MTH-06: Shadow / Highlight Field
        # --------------------------------------------------------------
        r06 = self.shadow_field.transform(current)
        current = r06.output_rgb

        # --------------------------------------------------------------
        # MTH-07: Character / Geometry Field
        # --------------------------------------------------------------
        r07 = self.geometry_field.transform(current, geom_obs)
        current = r07.output_rgb

        # --------------------------------------------------------------
        # MTH-08: Face / Facial Feature Field
        # --------------------------------------------------------------
        r08 = self.face_field.transform(current, geom_obs)
        current = r08.output_rgb

        # --------------------------------------------------------------
        # MTH-09: Cinematic Lighting Field
        # --------------------------------------------------------------
        r09 = self.lighting_field.transform(current, geom_obs)
        current = r09.output_rgb

        # --------------------------------------------------------------
        # MTH-10: Temporal Field
        # --------------------------------------------------------------
        r10 = self.temporal_field.transform(current, temporal_observation)

        # MTH-10 output_rgb is float32 [0.0, 1.0], convert to uint8
        output_uint8 = np.clip(
            r10.output_rgb * 255.0 + 0.5,
            0.0,
            255.0,
        ).astype(np.uint8)

        return MathematicalRenderResult(
            output_rgb=output_uint8,
            mth02=r02,
            mth03=r03,
            mth04=r04,
            mth05=r05,
            mth06=r06,
            mth07=r07,
            mth08=r08,
            mth09=r09,
            mth10=r10,
        )

    def render_video(
        self,
        input_path: str | Path,
        output_path: str | Path,
        vision_jsonl: Optional[str | Path] = None,
        temporal_plan: Optional[str | Path] = None,
        quality_report_path: Optional[str | Path] = None,
        control: Optional[Any] = None,
    ) -> Path:
        """
        Renders a full video frame-by-frame through MTH-02 → MTH-10.
        """
        in_path = Path(input_path).resolve()
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if not in_path.exists():
            raise FileNotFoundError(f"Input video not found: {in_path}")

        # Load Vision observations if provided
        vision_frames: Dict[int, Any] = {}
        if vision_jsonl and Path(vision_jsonl).exists():
            try:
                from src.vision.jsonl import read_vision_jsonl
                v_list = read_vision_jsonl(str(vision_jsonl))
                vision_frames = {vf.frame_index: vf for vf in v_list}
            except Exception:
                pass

        # Load Temporal Plan if provided
        scene_cut_frames: set[int] = set()
        if temporal_plan and Path(temporal_plan).exists():
            try:
                with open(temporal_plan, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            if record.get("scene_cut", False):
                                scene_cut_frames.add(record.get("frame_index", -1))
            except Exception:
                pass

        capture = cv2.VideoCapture(str(in_path))
        if not capture.isOpened():
            raise RenderingError(f"Could not open input video: {in_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RenderingError(f"Could not create output video writer: {out_path}")

        self.reset()
        processed_count = 0
        start_time = time.perf_counter()

        try:
            while True:
                if control is not None:
                    control.check()

                ok, frame_bgr = capture.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                # Determine scene cut
                is_scene_cut = (processed_count in scene_cut_frames)
                vf = vision_frames.get(processed_count)
                if vf is not None and getattr(vf, "scene_cut", False):
                    is_scene_cut = True

                # Optical flow calculation for temporal stabilization
                optical_flow = None
                if self._previous_input_gray is not None and not is_scene_cut:
                    optical_flow = cv2.calcOpticalFlowFarneback(
                        self._previous_input_gray,
                        frame_gray,
                        None,
                        0.5,
                        3,
                        15,
                        3,
                        5,
                        1.2,
                        0,
                    )

                temporal_obs = TemporalObservation(
                    optical_flow=optical_flow,
                    scene_cut=is_scene_cut,
                )

                try:
                    result = self.render(
                        frame_rgb=frame_rgb,
                        vision=vf,
                        temporal_observation=temporal_obs,
                    )
                except Exception as exc:
                    raise RenderingError(
                        f"Rendering failed at frame {processed_count}: {exc}"
                    ) from exc

                out_bgr = cv2.cvtColor(result.output_rgb, cv2.COLOR_RGB2BGR)
                writer.write(out_bgr)

                self._previous_input_gray = frame_gray.copy()
                processed_count += 1

                elapsed_so_far = time.perf_counter() - start_time
                curr_fps = processed_count / elapsed_so_far if elapsed_so_far > 0 else 0.0
                curr_eta = ((total_frames - processed_count) / curr_fps) if (curr_fps > 0 and total_frames > 0) else 0.0

                if control is not None and total_frames > 0:
                    control.update(
                        progress=processed_count / total_frames,
                        current_frame=processed_count,
                        total_frames=total_frames,
                        fps=round(curr_fps, 1),
                        eta_seconds=round(curr_eta, 1),
                        elapsed_seconds=round(elapsed_so_far, 1),
                        substage="MTH-09 Lighting Field",
                        message=f"Rendering frame {processed_count}/{total_frames}",
                    )

                if self.event_bus is not None and total_frames > 0:
                    try:
                        from src.core.events import PipelineEvent
                        job_obj = getattr(control, "job", None) if control else None
                        j_id = getattr(job_obj, "job_id", "") if job_obj else ""
                        self.event_bus.emit(
                            PipelineEvent(
                                job_id=j_id,
                                stage="artistic",
                                substage="MTH-09 Lighting Field",
                                progress=processed_count / total_frames,
                                frame=processed_count,
                                total_frames=total_frames,
                                fps=round(curr_fps, 1),
                                eta_seconds=round(curr_eta, 1),
                                message=f"Rendering frame {processed_count}/{total_frames} (MTH-02..10)",
                            )
                        )
                    except Exception:
                        pass

        finally:
            capture.release()
            writer.release()

        elapsed = time.perf_counter() - start_time
        speed_fps = processed_count / elapsed if elapsed > 0 else 0.0

        if quality_report_path:
            q_path = Path(quality_report_path)
            q_path.parent.mkdir(parents=True, exist_ok=True)
            report = {
                "input": str(in_path),
                "output": str(out_path),
                "frames_processed": processed_count,
                "total_frames": total_frames,
                "fps": fps,
                "runtime_seconds": round(elapsed, 2),
                "throughput_fps": round(speed_fps, 2),
                "style": getattr(self.style, "name", "creator_anime"),
                "status": "success",
            }
            with open(q_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return out_path
