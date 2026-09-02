"""Analyze a complete video using the Phase 2 Vision Engine."""
import argparse
import json
import sys
import time
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

# Ensure root dir in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.vision.vision_pipeline import VisionEngine
from src.io.video_io import inspect_video, create_video_writer


def analyze_video(
    video_path: str,
    output_dir: str = "analysis_output",
    max_frames: int = None,
    no_video: bool = False,
):
    video_p = Path(video_path)
    out_dir_p = Path(output_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if not video_p.exists():
        print(f"[ERROR] Video file not found: {video_p}")
        sys.exit(1)

    metadata = inspect_video(video_p)
    print("==================================================")
    print("    PHASE 2: VIDEO SCENE UNDERSTANDING ENGINE     ")
    print("==================================================")
    print(f"[*] Input:      {video_p.name}")
    print(f"[*] Resolution: {metadata.resolution_str} @ {metadata.fps:.2f} FPS")
    print(f"[*] Frames:     {metadata.frame_count} ({metadata.duration_sec:.2f}s)")
    print(f"[*] Audio:      {'Yes' if metadata.has_audio else 'No'}")
    print("--------------------------------------------------")

    cap = cv2.VideoCapture(str(video_p))
    total_frames = metadata.frame_count
    if max_frames and max_frames > 0:
        total_frames = min(total_frames, max_frames)

    jsonl_path = out_dir_p / "vision.jsonl"
    summary_path = out_dir_p / "summary.json"
    annotated_mp4_path = out_dir_p / "annotated.mp4"

    writer = None
    if not no_video:
        writer = create_video_writer(
            output_path=annotated_mp4_path,
            width=metadata.width,
            height=metadata.height,
            fps=metadata.fps,
            fourcc_str="mp4v",
        )

    engine = VisionEngine()

    jsonl_file = open(jsonl_path, "w", encoding="utf-8")

    # Aggregate Statistics
    stats = {
        "source_video": str(video_p.resolve()),
        "resolution": metadata.resolution_str,
        "fps": metadata.fps,
        "total_frames_analyzed": 0,
        "frames_with_face": 0,
        "frames_with_pose": 0,
        "frames_with_hands": 0,
        "total_motion_energy": 0.0,
        "mean_person_coverage": 0.0,
        "processing_time_sec": 0.0,
        "processing_fps": 0.0,
    }

    start_time = time.time()
    pbar = tqdm(total=total_frames, desc="Analyzing Video", unit="frame")

    frame_idx = 0
    while True:
        if max_frames and frame_idx >= max_frames:
            break

        ret, bgr = cap.read()
        if not ret:
            break

        timestamp = frame_idx / metadata.fps if metadata.fps > 0 else 0.0
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        vision_data, annotated_rgb = engine.process_frame(
            rgb=rgb,
            frame_index=frame_idx,
            timestamp=timestamp,
            generate_annotated=(not no_video),
        )

        # Write record to JSONL
        jsonl_file.write(json.dumps(vision_data.to_dict()) + "\n")

        # Write annotated frame
        if writer is not None and annotated_rgb is not None:
            writer.write(cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR))

        # Update stats
        stats["total_frames_analyzed"] += 1
        if len(vision_data.faces) > 0:
            stats["frames_with_face"] += 1
        if vision_data.pose is not None:
            stats["frames_with_pose"] += 1
        if len(vision_data.hands) > 0:
            stats["frames_with_hands"] += 1
        if vision_data.motion.valid:
            stats["total_motion_energy"] += vision_data.motion.mean_magnitude
        if vision_data.person_mask:
            stats["mean_person_coverage"] += vision_data.person_mask.coverage

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    jsonl_file.close()
    engine.close()
    if writer is not None:
        writer.release()

    elapsed = max(0.001, time.time() - start_time)
    stats["processing_time_sec"] = round(elapsed, 2)
    stats["processing_fps"] = round(stats["total_frames_analyzed"] / elapsed, 2)
    if stats["total_frames_analyzed"] > 0:
        stats["mean_person_coverage"] = round(
            stats["mean_person_coverage"] / stats["total_frames_analyzed"], 4
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("==================================================")
    print("          ANALYSIS SUMMARY & ARTIFACTS            ")
    print("==================================================")
    print(f"[+] Total Frames Analyzed: {stats['total_frames_analyzed']}")
    print(f"[+] Face Detection Rate:   {(stats['frames_with_face']/max(1,stats['total_frames_analyzed']))*100:.1f}%")
    print(f"[+] Pose Detection Rate:   {(stats['frames_with_pose']/max(1,stats['total_frames_analyzed']))*100:.1f}%")
    print(f"[+] Hands Detection Rate:  {(stats['frames_with_hands']/max(1,stats['total_frames_analyzed']))*100:.1f}%")
    print(f"[+] Processing Speed:      {stats['processing_fps']} FPS")
    print("--------------------------------------------------")
    print(f"[SUCCESS] JSONL Database:    {jsonl_path}")
    print(f"[SUCCESS] Summary Report:    {summary_path}")
    if not no_video:
        print(f"[SUCCESS] Annotated Debug:   {annotated_mp4_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Video Scene Understanding Engine")
    parser.add_argument("video", type=str, help="Path to input video (.mp4/.mov)")
    parser.add_argument("--output-dir", "-o", type=str, default="analysis_output", help="Directory for output files")
    parser.add_argument("--max-frames", "-m", type=int, default=None, help="Max frames to analyze")
    parser.add_argument("--no-video", action="store_true", help="Skip generating debug annotated.mp4")

    args = parser.parse_args()
    analyze_video(args.video, args.output_dir, args.max_frames, args.no_video)


if __name__ == "__main__":
    main()
