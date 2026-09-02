import json
import cv2
import numpy as np

from src.art.video_renderer import VideoRenderer


def create_test_video(path, frames=12):
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (160, 120),
    )

    for i in range(frames):
        frame = np.zeros(
            (120, 160, 3),
            dtype=np.uint8,
        )
        x = 20 + i * 5
        cv2.circle(
            frame,
            (x, 60),
            20,
            (255, 255, 255),
            -1,
        )
        writer.write(frame)

    writer.release()


def create_vision_jsonl(path, frames=12):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:
        for i in range(frames):
            record = {
                "frame_index": i,
                "timestamp": i / 30.0,
                "scene_id": 0,
                "scene_cut": i == 0,
            }
            handle.write(
                json.dumps(record)
                + "\n"
            )


def test_complete_art_video_pipeline(tmp_path):
    input_video = tmp_path / "input.mp4"
    vision_jsonl = tmp_path / "vision.jsonl"
    output_video = tmp_path / "output.mp4"

    create_test_video(
        str(input_video)
    )

    create_vision_jsonl(
        str(vision_jsonl)
    )

    renderer = VideoRenderer()
    result = renderer.render(
        str(input_video),
        str(vision_jsonl),
        str(output_video),
    )

    assert result["frames"] == 12
    assert output_video.exists()

    capture = cv2.VideoCapture(
        str(output_video)
    )

    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1

    capture.release()
    assert count == 12
