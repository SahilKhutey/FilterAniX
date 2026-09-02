"""Consistency Auditor and Quality Reporting Engine."""
import json
from pathlib import Path
from typing import Optional
import cv2
from tqdm import tqdm

from src.consistency.types import ReferenceProfile, ConsistencyReport
from src.consistency.identity import IdentityScorer
from src.io.video_io import inspect_video


class ConsistencyAuditor:
    """Evaluates generated video against a reference profile and produces consistency_report.json."""

    def __init__(self, profile: ReferenceProfile, warning_threshold: float = 0.55):
        self.profile = profile
        self.scorer = IdentityScorer(profile, warning_threshold=warning_threshold)

    def audit_video(
        self,
        video_path: str | Path,
        output_report_path: str | Path = "consistency_report.json",
        max_frames: Optional[int] = None,
    ) -> ConsistencyReport:
        video_p = Path(video_path)
        out_p = Path(output_report_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        metadata = inspect_video(video_p)
        total_frames = metadata.frame_count
        if max_frames and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        cap = cv2.VideoCapture(str(video_p))
        scores = []
        warning_count = 0

        pbar = tqdm(total=total_frames, desc="Auditing Consistency", unit="frame")

        for _ in range(total_frames):
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            metrics = self.scorer.evaluate_frame(rgb)

            scores.append(metrics.similarity)
            if metrics.warning:
                warning_count += 1

            pbar.update(1)

        pbar.close()
        cap.release()

        mean_sim = float(sum(scores) / len(scores)) if scores else 0.0
        min_sim = float(min(scores)) if scores else 0.0
        max_sim = float(max(scores)) if scores else 0.0
        warning_ratio = float(warning_count) / float(len(scores)) if scores else 0.0

        report = ConsistencyReport(
            frames=len(scores),
            fps=metadata.fps,
            duration_seconds=len(scores) / metadata.fps if metadata.fps > 0 else 0.0,
            mean_similarity=mean_sim,
            minimum_similarity=min_sim,
            maximum_similarity=max_sim,
            warning_frame_count=warning_count,
            warning_ratio=warning_ratio,
            frame_scores=scores,
        )

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
