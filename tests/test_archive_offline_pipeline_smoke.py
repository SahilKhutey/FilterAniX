"""Synthetic Test Video Generator & Smoke Test Suite."""
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

# Ensure archive directory is in sys.path
ARCHIVE_DIR = Path(__file__).resolve().parent.parent / "archive"
if str(ARCHIVE_DIR) not in sys.path:
    sys.path.insert(0, str(ARCHIVE_DIR))

from filteranix.core.config import load_config
from filteranix.pipeline.offline_pipeline import OfflineVideoPipeline


def generate_synthetic_creator_video(
    output_path: Path | str, num_frames: int = 60, width: int = 640, height: int = 360, fps: int = 30
):
    """Generates a synthetic creator video with static room, desk, mic, and moving person."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for t in range(num_frames):
        # 1. Static Room Background (Wall, Poster, Bookshelf)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [190, 205, 215]  # Soft light blue-gray wall

        # Wall Poster
        cv2.rectangle(frame, (50, 40), (160, 150), (140, 110, 80), -1)
        cv2.rectangle(frame, (55, 45), (155, 145), (230, 210, 180), -1)
        cv2.circle(frame, (105, 95), 25, (100, 160, 220), -1)

        # Bookshelf on the right
        cv2.rectangle(frame, (480, 30), (600, 220), (80, 50, 30), -1)
        for shelf_y in [80, 140, 200]:
            cv2.line(frame, (480, shelf_y), (600, shelf_y), (60, 35, 20), 4)
            # Books
            cv2.rectangle(frame, (490, shelf_y - 45), (510, shelf_y), (180, 70, 60), -1)
            cv2.rectangle(frame, (520, shelf_y - 50), (545, shelf_y), (70, 140, 90), -1)
            cv2.rectangle(frame, (555, shelf_y - 40), (580, shelf_y), (210, 180, 50), -1)

        # 2. Moving Person (Sitting at center desk)
        head_drift_x = int(12 * math.sin(t * 0.12))
        head_drift_y = int(6 * math.cos(t * 0.10))
        cx = width // 2 + head_drift_x
        cy = height // 2 - 20 + head_drift_y

        # Body / Shoulders (Dark Navy T-shirt)
        body_pts = np.array([
            [cx - 90, height - 30],
            [cx + 90, height - 30],
            [cx + 70, cy + 80],
            [cx - 70, cy + 80]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [body_pts], (45, 40, 60))

        # Neck
        cv2.rectangle(frame, (cx - 18, cy + 45), (cx + 18, cy + 85), (200, 175, 155), -1)

        # Head / Face (Skin tone)
        cv2.ellipse(frame, (cx, cy), (48, 62), 0, 0, 360, (220, 195, 175), -1)

        # Hair (Anime styled top & bangs)
        hair_pts = np.array([
            [cx - 52, cy - 10],
            [cx - 55, cy - 55],
            [cx - 20, cy - 72],
            [cx + 25, cy - 70],
            [cx + 55, cy - 50],
            [cx + 50, cy - 10],
            [cx + 35, cy - 35],
            [cx, cy - 40],
            [cx - 30, cy - 35]
        ], dtype=np.int32)
        cv2.fillPoly(frame, [hair_pts], (35, 30, 45))

        # Eyes (Blinking every 20 frames)
        eye_open = 1 if (t % 25 < 22) else 0
        if eye_open:
            cv2.circle(frame, (cx - 18, cy - 5), 6, (40, 30, 30), -1)
            cv2.circle(frame, (cx + 18, cy - 5), 6, (40, 30, 30), -1)
            # Catchlight
            cv2.circle(frame, (cx - 16, cy - 7), 2, (255, 255, 255), -1)
            cv2.circle(frame, (cx + 20, cy - 7), 2, (255, 255, 255), -1)
        else:
            cv2.line(frame, (cx - 24, cy - 5), (cx - 12, cy - 5), (40, 30, 30), 2)
            cv2.line(frame, (cx + 12, cy - 5), (cx + 24, cy - 5), (40, 30, 30), 2)

        # Mouth (Moving subtly while talking)
        mouth_h = int(2 + 4 * abs(math.sin(t * 0.35)))
        cv2.ellipse(frame, (cx, cy + 28), (8, mouth_h), 0, 0, 360, (180, 80, 80), -1)

        # Moving Hand (Moving from Left to Right across frames 10 to 50)
        hand_progress = np.clip((t - 10) / 40.0, 0.0, 1.0)
        hand_x = int((width // 2 - 120) + hand_progress * 240)
        hand_y = int(height // 2 + 60 - 25 * math.sin(hand_progress * math.pi))
        cv2.circle(frame, (hand_x, hand_y), 18, (220, 195, 175), -1)
        # Fingers
        for finger_i in range(4):
            fx = hand_x - 12 + finger_i * 8
            cv2.line(frame, (fx, hand_y), (fx, hand_y - 14), (210, 185, 165), 4)

        # 3. Desk Foreground (Laptop & Microphone)
        desk_y = height - 70
        cv2.rectangle(frame, (0, desk_y), (width, height), (75, 55, 45), -1)

        # Laptop (Right side of desk)
        cv2.rectangle(frame, (380, desk_y - 45), (490, desk_y + 10), (130, 135, 140), -1)
        cv2.rectangle(frame, (385, desk_y - 40), (485, desk_y), (60, 120, 180), -1)

        # Creator Microphone (Left/Center foreground)
        cv2.circle(frame, (180, desk_y - 30), 22, (50, 50, 55), -1)
        cv2.rectangle(frame, (170, desk_y - 30), (190, desk_y + 20), (40, 40, 45), -1)
        cv2.line(frame, (180, desk_y + 20), (180, desk_y + 55), (30, 30, 35), 6)

        writer.write(frame)

    writer.release()
    return output_path


def test_end_to_end_pipeline(tmp_path):
    """Verifies that the offline pipeline runs smoothly end-to-end on synthetic footage."""
    test_video = tmp_path / "test_input.mp4"
    output_video = tmp_path / "test_output.mp4"

    generate_synthetic_creator_video(test_video, num_frames=20, width=320, height=180)
    assert test_video.exists()

    config = load_config(
        pipeline_path="configs/default_pipeline.yaml",
        style_path="configs/styles/creator_anime.yaml",
        character_path="configs/characters/creator_default.yaml",
    )
    config.pipeline.target_width = 320
    config.pipeline.target_height = 180

    pipeline = OfflineVideoPipeline(config)
    pipeline.process_video(input_path=test_video, output_path=output_video, side_by_side=True)

    assert output_video.exists()
    assert output_video.stat().st_size > 1000
