"""Video-Level Artistic Style Renderer with JSONL Vision Integration and FFmpeg Audio Sync."""
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional
import cv2
import numpy as np
from tqdm import tqdm

from src.art.types import RenderConfig
from src.art.style_engine import StyleEngine
from src.core.models import ProcessingProgress
from src.io.video_io import inspect_video, create_video_writer, merge_audio_and_video
from src.vision.models import FrameVisionData, FaceData, PoseData, HandData, Landmark, BoundingBox, PersonMaskData, MotionData


class VideoStyleRenderer:
    """Processes entire videos through the Phase 3 Artistic Style Engine."""

    def __init__(self, config: Optional[RenderConfig] = None):
        self.config = config or RenderConfig()
        self.style_engine = StyleEngine(self.config)

    def _load_vision_jsonl(self, jsonl_path: Optional[str | Path]) -> Dict[int, FrameVisionData]:
        """Loads and parses vision.jsonl into a frame-indexed dictionary of FrameVisionData."""
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

                # Reconstruct models
                faces = []
                for f_d in d.get("faces", []):
                    bbox = BoundingBox(**f_d["bbox"])
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
                    bbox = BoundingBox(**p_d["bbox"])
                    lms = [Landmark(**lm) for lm in p_d.get("landmarks", [])]
                    pose = PoseData(
                        landmarks=lms,
                        bbox=bbox,
                        landmark_count=p_d.get("landmark_count", len(lms)),
                        torso_center=p_d.get("torso_center"),
                    )

                hands = []
                for h_d in d.get("hands", []):
                    bbox = BoundingBox(**h_d["bbox"])
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
    ) -> str:
        """Executes full video stylization with temporal stabilization and audio preservation."""
        input_path = Path(input_path).resolve()
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = inspect_video(input_path)
        total_frames = metadata.frame_count
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Load Vision Data
        vision_map = self._load_vision_jsonl(vision_jsonl)

        # Load Reference Image
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

            # Render artistic frame
            art_rgb = self.style_engine.render_frame(
                rgb=rgb,
                vision_data=vision_data,
                reference_rgb=ref_rgb,
                stabilize=True,
            )

            # Composite side-by-side or solo
            if side_by_side:
                combined = np.hstack([rgb, art_rgb])
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            else:
                writer.write(cv2.cvtColor(art_rgb, cv2.COLOR_RGB2BGR))

            # Telemetry
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

        # Mux with original audio via FFmpeg
        merge_audio_and_video(
            silent_video_path=temp_silent_video,
            audio_source_path=input_path,
            final_output_path=output_path,
            has_audio=metadata.has_audio,
        )

        if temp_silent_video.exists():
            temp_silent_video.unlink(missing_ok=True)

        return str(output_path)
