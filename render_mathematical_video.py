"""FilterAniX Mathematical Anime Engine Video Renderer CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from tqdm import tqdm

# Ensure root in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.art.mathematical import (
    MathematicalAnimeEngine,
    MathematicalAnimeStyle,
    DEFAULT_ANIME_PALETTE,
)
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


def load_vision_jsonl(jsonl_path: Path | str | None) -> dict[int, FrameVisionData]:
    if not jsonl_path or not Path(jsonl_path).exists():
        return {}

    vision_map = {}
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
                        mouth_opening=float(f_d.get("mouth_opening", f_d.get("mouth_open", 0.0))),
                        left_eye_ear=float(f_d.get("left_eye_ear", 0.0)),
                        right_eye_ear=float(f_d.get("right_eye_ear", 0.0)),
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
                        confidence=float(h_d.get("confidence", 0.9)),
                        landmarks=lms,
                        bbox=bbox,
                    )
                )

            person_mask = None
            if d.get("person_mask"):
                m_d = d["person_mask"]
                bbox = BoundingBox(**m_d["bbox"]) if m_d.get("bbox") else None
                person_mask = PersonMaskData(
                    threshold=float(m_d.get("threshold", 0.5)),
                    coverage=float(m_d.get("coverage", 0.0)),
                    bbox=bbox,
                )

            motion = MotionData(**d.get("motion", {})) if "motion" in d else MotionData()

            frame_vision = FrameVisionData(
                frame_index=idx,
                timestamp=d.get("timestamp", 0.0),
                width=d.get("width", 1280),
                height=d.get("height", 720),
                faces=faces,
                pose=pose,
                hands=hands,
                person_mask=person_mask,
                motion=motion,
                objects=[],
            )
            vision_map[idx] = frame_vision

    return vision_map


def render_mathematical_video(
    video_path: str | Path,
    output_path: str | Path,
    vision_jsonl: str | Path | None = None,
    max_frames: int | None = None,
    side_by_side: bool = False,
    contrast: float = 1.08,
    gamma: float = 0.96,
    palette_mix: float = 0.60,
    edge_strength: float = 0.72,
    shadow_strength: float = 0.20,
    highlight_strength: float = 0.10,
    skin_smoothing: float = 0.70,
    temporal_strength: float = 0.12,
    use_optical_flow: bool = True,
) -> dict:
    video_p = Path(video_path).resolve()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not video_p.exists():
        raise FileNotFoundError(f"Input video not found: {video_p}")

    meta = inspect_video(video_p)
    total_frames = meta.frame_count
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    style = MathematicalAnimeStyle(
        contrast=contrast,
        gamma=gamma,
        palette_mix=palette_mix,
        edge_strength=edge_strength,
        shadow_strength=shadow_strength,
        highlight_strength=highlight_strength,
        skin_smoothing=skin_smoothing,
        temporal_strength=temporal_strength,
        use_optical_flow=use_optical_flow,
    )
    engine = MathematicalAnimeEngine(style)

    vision_map = load_vision_jsonl(vision_jsonl)

    print("==================================================")
    print("  FilterAniX — Mathematical Anime Engine v1.0    ")
    print("==================================================")
    print(f"[*] Input Video:    {video_p.name}")
    print(f"[*] Resolution:     {meta.width}x{meta.height} @ {meta.fps:.2f} FPS")
    print(f"[*] Frames to Run:  {total_frames} ({total_frames / meta.fps:.2f}s)")
    print(f"[*] Vision Data:    {len(vision_map)} frames loaded" if vision_map else "[*] Vision Data:    None (procedural prior)")
    print(f"[*] Audio Present:  {'Yes' if meta.has_audio else 'No'}")
    print(f"[*] Mode:           Every frame / Every pixel recalculated (Deterministic)")
    print("--------------------------------------------------")

    temp_silent = out_p.parent / f"temp_math_silent_{int(time.time()*1000)}.mp4"
    out_w = meta.width * 2 if side_by_side else meta.width
    out_h = meta.height

    writer = create_video_writer(
        output_path=temp_silent,
        width=out_w,
        height=out_h,
        fps=meta.fps,
        fourcc_str="mp4v",
    )

    cap = cv2.VideoCapture(str(video_p))
    engine.reset()

    start_time = time.time()
    rendered_frames = 0

    pbar = tqdm(total=total_frames, desc="Transforming Fields", unit="frame")

    try:
        for idx in range(total_frames):
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            v_data = vision_map.get(idx, None)
            is_scene_cut = (idx == 0)

            # Mathematical Anime Transformation
            art_rgb = engine.render(
                rgb=rgb,
                vision_data=v_data,
                scene_cut=is_scene_cut,
                stabilize=True,
            )

            if side_by_side:
                combined = np.hstack([rgb, art_rgb])
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            else:
                writer.write(cv2.cvtColor(art_rgb, cv2.COLOR_RGB2BGR))

            rendered_frames += 1
            pbar.update(1)
    finally:
        pbar.close()
        cap.release()
        writer.release()

    elapsed = max(0.001, time.time() - start_time)
    avg_fps = rendered_frames / elapsed

    print("--------------------------------------------------")
    print("[*] Muxing and syncing audio track...")
    merge_audio_and_video(
        silent_video_path=temp_silent,
        audio_source_path=video_p,
        final_output_path=out_p,
        has_audio=meta.has_audio,
    )

    if temp_silent.exists():
        temp_silent.unlink(missing_ok=True)

    summary = engine.diagnostics.summarize()

    report = {
        "input_video": str(video_p),
        "output_video": str(out_p),
        "input_frames": meta.frame_count,
        "rendered_frames": rendered_frames,
        "width": out_w,
        "height": out_h,
        "fps": meta.fps,
        "audio_preserved": meta.has_audio,
        "total_seconds": round(elapsed, 2),
        "processing_fps": round(avg_fps, 2),
        "average_frame_latency_ms": summary.average_latency_ms,
        "p95_frame_latency_ms": summary.p95_latency_ms,
    }

    print(f"[SUCCESS] Anime render complete -> {out_p}")
    print(f"[*] Processed {rendered_frames}/{total_frames} frames in {elapsed:.1f}s ({avg_fps:.2f} FPS)")

    return report


def main():
    parser = argparse.ArgumentParser(description="FilterAniX Mathematical Anime Engine v1.0 CLI")
    parser.add_argument("--video", "-v", required=True, help="Path to input video (.mp4/.mov)")
    parser.add_argument("--output", "-o", default="output_math_anime.mp4", help="Output path")
    parser.add_argument("--vision-jsonl", "-j", default=None, help="Optional Phase 2 vision.jsonl")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Frame cap")
    parser.add_argument("--side-by-side", action="store_true", help="Render side-by-side comparison")

    args = parser.parse_args()
    report = render_mathematical_video(
        video_path=args.video,
        output_path=args.output,
        vision_jsonl=args.vision_jsonl,
        max_frames=args.max_frames,
        side_by_side=args.side_by_side,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
