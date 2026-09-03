from pathlib import Path
import json
import numpy as np
import cv2

from src.art.keyframe_video_renderer import KeyframeVideoRenderer
from src.art.types import StyleConfig


def test_keyframe_video_renderer_execution(tmp_path):
    input_video = tmp_path / "test_input.mp4"
    output_video = tmp_path / "artistic" / "animated.mp4"
    temporal_plan_path = tmp_path / "temporal_plan.jsonl"
    vision_jsonl_path = tmp_path / "vision.jsonl"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(input_video), fourcc, 30, (160, 120))
    for i in range(12):
        frame = np.full((120, 160, 3), 40 + i * 3, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with open(temporal_plan_path, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(json.dumps({
                "frame_index": i,
                "scene_id": 0,
                "is_keyframe": (i % 6 == 0),
                "is_scene_cut": (i == 0),
                "motion_score": 0.05,
                "reference_strength": 0.55,
            }) + "\n")

    with open(vision_jsonl_path, "w", encoding="utf-8") as f:
        for i in range(12):
            f.write(json.dumps({
                "frame_index": i,
                "faces": [],
            }) + "\n")

    renderer = KeyframeVideoRenderer(config=StyleConfig(keyframe_interval=6))
    res = renderer.render_video(
        input_path=str(input_video),
        vision_jsonl=str(vision_jsonl_path),
        temporal_plan=str(temporal_plan_path),
        output_path=str(output_video),
    )

    assert output_video.exists()
    assert res["frames"] == 12
    assert res["keyframes"] >= 2
    assert (tmp_path / "artistic" / "keyframes").exists()
    assert (tmp_path / "artistic" / "render_metrics.json").exists()
