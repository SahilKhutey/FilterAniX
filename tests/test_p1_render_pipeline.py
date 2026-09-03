from pathlib import Path
import json
import pytest
import numpy as np
import cv2

from src.art.video_renderer import VideoRenderer
from src.art.types import StyleConfig, RendererBackend
from src.art.render_metrics import RenderMetrics


def test_p1_artistic_outputs_exist(tmp_path):
    artistic_dir = tmp_path / "artistic"
    artistic_dir.mkdir(parents=True)
    keyframe_dir = artistic_dir / "keyframes"
    keyframe_dir.mkdir()

    metrics = artistic_dir / "render_metrics.json"
    metrics_obj = RenderMetrics(
        total_frames=30,
        keyframes=3,
        propagated_frames=27,
        diffusion_frames=0,
        fallback_frames=3,
        scene_cuts=1,
        render_seconds=1.2,
    )
    metrics_obj.save(metrics)

    assert keyframe_dir.exists()
    assert metrics.exists()

    with open(metrics, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_frames"] == 30
        assert data["keyframes"] == 3
        assert "keyframe_ratio" in data


def test_video_renderer_p1_execution(tmp_path):
    # Create short 15-frame video
    input_video = tmp_path / "test_input.mp4"
    output_video = tmp_path / "artistic" / "animated.mp4"
    temporal_plan_path = tmp_path / "temporal_plan.jsonl"
    vision_jsonl_path = tmp_path / "vision.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(input_video), fourcc, 30, (160, 120))
    for i in range(15):
        frame = np.full((120, 160, 3), 50 + i * 2, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # Create dummy vision and temporal plan
    with open(temporal_plan_path, "w", encoding="utf-8") as f:
        for i in range(15):
            f.write(json.dumps({
                "frame_index": i,
                "scene_id": 0,
                "is_keyframe": (i % 6 == 0),
                "is_scene_cut": (i == 0),
                "motion_score": 0.05,
                "reference_strength": 0.55,
            }) + "\n")

    with open(vision_jsonl_path, "w", encoding="utf-8") as f:
        for i in range(15):
            f.write(json.dumps({
                "frame_index": i,
                "faces": [],
            }) + "\n")

    renderer = VideoRenderer(config=StyleConfig(keyframe_interval=6, backend=RendererBackend.DIFFUSERS))
    res = renderer.render_video(
        video_path=str(input_video),
        vision_jsonl_path=str(vision_jsonl_path),
        output_path=str(output_video),
        temporal_plan_jsonl=str(temporal_plan_path),
    )

    assert output_video.exists()
    assert output_video.stat().st_size > 0
    assert (tmp_path / "artistic" / "keyframes").exists()
    assert (tmp_path / "artistic" / "render_metrics.json").exists()

    with open(tmp_path / "artistic" / "render_metrics.json", "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
        assert metrics_data["total_frames"] == 15
        assert metrics_data["keyframes"] >= 2
        assert metrics_data["propagated_frames"] > 0
