"""Synthetic Humanoid Creator Video Generator for Vision Pipeline & Regression Testing."""
import math
from pathlib import Path
import cv2
import numpy as np


def generate_synthetic_creator_video(
    output_path: Path | str,
    num_frames: int = 60,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
) -> Path:
    """Generates a synthetic creator video with static room, desk, mic, laptop, and moving humanoid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for t in range(num_frames):
        # 1. Static Studio Room Background (Wall, Poster, Bookshelf)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = [210, 218, 225]  # Soft neutral studio wall

        # Wall Poster
        cv2.rectangle(frame, (40, 30), (150, 140), (140, 110, 80), -1)
        cv2.rectangle(frame, (45, 35), (145, 135), (230, 210, 180), -1)
        cv2.circle(frame, (95, 85), 22, (100, 160, 220), -1)

        # Bookshelf on the right
        cv2.rectangle(frame, (490, 20), (610, 210), (80, 50, 30), -1)
        for shelf_y in [75, 135, 195]:
            cv2.line(frame, (490, shelf_y), (610, shelf_y), (60, 35, 20), 4)
            cv2.rectangle(frame, (500, shelf_y - 42), (520, shelf_y), (180, 70, 60), -1)
            cv2.rectangle(frame, (530, shelf_y - 46), (555, shelf_y), (70, 140, 90), -1)
            cv2.rectangle(frame, (565, shelf_y - 38), (590, shelf_y), (210, 180, 50), -1)

        # 2. Moving Humanoid Creator (Sitting at center desk)
        head_drift_x = int(10 * math.sin(t * 0.10))
        head_drift_y = int(5 * math.cos(t * 0.08))
        cx = width // 2 + head_drift_x
        cy = height // 2 - 25 + head_drift_y

        # Body / Shoulders (Navy studio shirt)
        body_pts = np.array([
            [cx - 110, height - 30],
            [cx + 110, height - 30],
            [cx + 85, cy + 85],
            [cx - 85, cy + 85],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [body_pts], (45, 40, 65))

        # Collar & Neck
        cv2.rectangle(frame, (cx - 20, cy + 50), (cx + 20, cy + 90), (195, 165, 145), -1)
        collar_pts = np.array([[cx - 30, cy + 85], [cx + 30, cy + 85], [cx, cy + 110]], dtype=np.int32)
        cv2.fillPoly(frame, [collar_pts], (230, 230, 235))

        # Head / Face Oval (Natural skin tone with slight shading)
        cv2.ellipse(frame, (cx, cy), (52, 70), 0, 0, 360, (190, 165, 145), -1)

        # Hair (Anime styled bangs & silhouette)
        hair_pts = np.array([
            [cx - 56, cy - 10],
            [cx - 60, cy - 60],
            [cx - 22, cy - 78],
            [cx + 28, cy - 75],
            [cx + 60, cy - 55],
            [cx + 55, cy - 10],
            [cx + 38, cy - 38],
            [cx, cy - 42],
            [cx - 32, cy - 38],
        ], dtype=np.int32)
        cv2.fillPoly(frame, [hair_pts], (35, 30, 45))

        # Eyebrows
        cv2.ellipse(frame, (cx - 22, cy - 22), (15, 4), -8, 0, 360, (30, 25, 35), -1)
        cv2.ellipse(frame, (cx + 22, cy - 22), (15, 4), 8, 0, 360, (30, 25, 35), -1)

        # Eyes with natural sclera, iris, pupil & catchlight
        eye_open = 1 if (t % 28 < 24) else 0  # Natural blink cycle
        if eye_open:
            # White sclera
            cv2.ellipse(frame, (cx - 22, cy - 8), (13, 8), 0, 0, 360, (250, 250, 252), -1)
            cv2.ellipse(frame, (cx + 22, cy - 8), (13, 8), 0, 0, 360, (250, 250, 252), -1)
            # Dark iris
            cv2.circle(frame, (cx - 22, cy - 8), 6, (55, 40, 35), -1)
            cv2.circle(frame, (cx + 22, cy - 8), 6, (55, 40, 35), -1)
            # Black pupil
            cv2.circle(frame, (cx - 22, cy - 8), 3, (15, 10, 10), -1)
            cv2.circle(frame, (cx + 22, cy - 8), 3, (15, 10, 10), -1)
            # Specular catchlight
            cv2.circle(frame, (cx - 20, cy - 10), 2, (255, 255, 255), -1)
            cv2.circle(frame, (cx + 24, cy - 10), 2, (255, 255, 255), -1)
        else:
            # Closed eye contour during blink
            cv2.line(frame, (cx - 30, cy - 8), (cx - 14, cy - 8), (35, 30, 30), 2)
            cv2.line(frame, (cx + 14, cy - 8), (cx + 30, cy - 8), (35, 30, 30), 2)

        # Nose bridge & nostrils
        cv2.line(frame, (cx, cy - 5), (cx - 3, cy + 16), (160, 135, 120), 2)
        cv2.ellipse(frame, (cx, cy + 18), (8, 5), 0, 0, 360, (160, 135, 120), -1)
        cv2.circle(frame, (cx - 4, cy + 19), 2, (80, 60, 50), -1)
        cv2.circle(frame, (cx + 4, cy + 19), 2, (80, 60, 50), -1)

        # Dynamic Mouth (Viseme speech movement)
        mouth_open = int(2 + 6 * abs(math.sin(t * 0.35)))
        cv2.ellipse(frame, (cx, cy + 38), (14, mouth_open + 2), 0, 0, 360, (140, 80, 95), -1)
        if mouth_open > 3:
            cv2.ellipse(frame, (cx, cy + 38), (10, mouth_open - 2), 0, 0, 360, (50, 15, 25), -1)
            cv2.rectangle(frame, (cx - 6, cy + 35), (cx + 6, cy + 37), (240, 240, 240), -1)

        # 3. Creator Gesturing Hand (Moving across frames 8 to 52)
        hand_progress = np.clip((t - 8) / 44.0, 0.0, 1.0)
        hand_x = int((width // 2 - 130) + hand_progress * 260)
        hand_y = int(height // 2 + 65 - 30 * math.sin(hand_progress * math.pi))
        
        # Arm connecting to body
        cv2.line(frame, (cx - 85, cy + 85), (hand_x, hand_y), (45, 40, 65), 18)
        # Palm
        cv2.ellipse(frame, (hand_x, hand_y), (16, 20), 0, 0, 360, (190, 165, 145), -1)
        # 5 Articulated Fingers
        for finger_i in range(5):
            fx = hand_x - 14 + finger_i * 7
            fy_tip = hand_y - 18 - (4 if finger_i in [1, 2] else 0)
            cv2.line(frame, (fx, hand_y), (fx, fy_tip), (180, 155, 135), 4)
            cv2.circle(frame, (fx, fy_tip), 2, (180, 155, 135), -1)

        # 4. Desk Foreground (Laptop & Studio Microphone)
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


if __name__ == "__main__":
    out = Path("tests/fixtures/creator_test_video.mp4")
    print(f"Generating synthetic creator fixture at {out}...")
    generate_synthetic_creator_video(out, num_frames=60)
    print("Generation complete!")
