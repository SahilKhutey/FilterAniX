"""Temporal Plan Generator (Produces temporal_plan.jsonl)."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from tqdm import tqdm

from src.consistency.types import RenderDecision, ReferenceProfile
from src.consistency.controller import TemporalController
from src.io.video_io import inspect_video
from src.vision.models import MotionData


class TemporalPlanner:
    """Precomputes and exports temporal_plan.jsonl for deterministic, drift-free rendering."""

    def __init__(self, keyframe_interval: int = 12, reference_profile: Optional[ReferenceProfile] = None):
        self.controller = TemporalController(
            keyframe_interval=keyframe_interval,
            reference_profile=reference_profile,
        )

    def generate_plan(
        self,
        video_path: str | Path,
        vision_jsonl_path: Optional[str | Path] = None,
        output_plan_path: str | Path = "temporal_plan.jsonl",
        max_frames: Optional[int] = None,
    ) -> str:
        video_p = Path(video_path)
        output_p = Path(output_plan_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        metadata = inspect_video(video_p)
        total_frames = metadata.frame_count
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        # Load Motion Data from vision.jsonl if available
        motion_map: Dict[int, MotionData] = {}
        if vision_jsonl_path and Path(vision_jsonl_path).exists():
            with open(vision_jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    idx = d.get("frame_index", 0)
                    if "motion" in d:
                        motion_map[idx] = MotionData(**d["motion"])

        cap = cv2.VideoCapture(str(video_p))
        self.controller.reset()

        with open(output_p, "w", encoding="utf-8") as out_f:
            pbar = tqdm(total=total_frames, desc="Building Temporal Plan", unit="frame")
            for frame_idx in range(total_frames):
                ret, bgr = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                timestamp = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
                motion_data = motion_map.get(frame_idx, None)

                decision: RenderDecision = self.controller.evaluate_frame(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    frame_rgb=rgb,
                    motion_data=motion_data,
                )

                out_f.write(json.dumps(decision.to_dict()) + "\n")
                pbar.update(1)

            pbar.close()
        cap.release()

        return str(output_p)
