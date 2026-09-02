from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 480
FPS = 30
FRAMES = 180


def ellipse(
    frame: np.ndarray,
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = -1,
) -> None:
    cv2.ellipse(
        frame,
        center,
        axes,
        angle,
        0,
        360,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_person(frame: np.ndarray, index: int) -> None:
    t = index / FPS

    # Small body sway.
    sway = int(8 * math.sin(t * 2.0))
    cx = WIDTH // 2 + sway

    # -------------------------
    # Body & Shoulders (Navy Shirt: BGR 160, 105, 80)
    # -------------------------
    cv2.ellipse(
        frame,
        (cx, 355),
        (95, 115),
        0,
        0,
        360,
        (160, 105, 80),
        -1,
        cv2.LINE_AA,
    )

    cv2.line(
        frame,
        (cx - 75, 330),
        (cx + 75, 330),
        (160, 105, 80),
        35,
        cv2.LINE_AA,
    )

    # -------------------------
    # Neck (Skin: BGR 145, 165, 215)
    # -------------------------
    cv2.rectangle(
        frame,
        (cx - 24, 250),
        (cx + 24, 305),
        (145, 165, 215),
        -1,
    )

    # -------------------------
    # Head & Face
    # -------------------------
    head_y = 185 + int(3 * math.sin(t * 1.5))

    ellipse(
        frame,
        (cx, head_y),
        (75, 95),
        color=(145, 165, 215),  # Natural skin tone in BGR
    )

    # Hair (Dark Brown: BGR 35, 30, 45)
    cv2.ellipse(
        frame,
        (cx, head_y - 42),
        (78, 60),
        0,
        180,
        360,
        (35, 30, 45),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        frame,
        (cx - 78, head_y - 45),
        (cx - 58, head_y + 30),
        (35, 30, 45),
        -1,
    )
    cv2.rectangle(
        frame,
        (cx + 58, head_y - 45),
        (cx + 78, head_y + 30),
        (35, 30, 45),
        -1,
    )

    # Eyebrows
    cv2.ellipse(
        frame,
        (cx - 28, head_y - 25),
        (18, 5),
        -8,
        0,
        360,
        (30, 25, 35),
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        frame,
        (cx + 28, head_y - 25),
        (18, 5),
        8,
        0,
        360,
        (30, 25, 35),
        -1,
        cv2.LINE_AA,
    )

    # -------------------------
    # Eyes & Blinking
    # -------------------------
    blink_phase = index % 90
    if 42 <= blink_phase <= 48:
        # Closed eyes during blink
        cv2.line(
            frame,
            (cx - 40, head_y - 8),
            (cx - 16, head_y - 8),
            (35, 25, 20),
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            frame,
            (cx + 16, head_y - 8),
            (cx + 40, head_y - 8),
            (35, 25, 20),
            4,
            cv2.LINE_AA,
        )
    else:
        # Open eyes (Sclera + Iris + Pupil + Catchlight)
        ellipse(
            frame,
            (cx - 28, head_y - 8),
            (15, 9),
            color=(250, 250, 252),
        )
        ellipse(
            frame,
            (cx + 28, head_y - 8),
            (15, 9),
            color=(250, 250, 252),
        )
        cv2.circle(frame, (cx - 28, head_y - 8), 6, (40, 35, 45), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + 28, head_y - 8), 6, (40, 35, 45), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 28, head_y - 8), 3, (15, 10, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + 28, head_y - 8), 3, (15, 10, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx - 26, head_y - 10), 2, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + 30, head_y - 10), 2, (255, 255, 255), -1, cv2.LINE_AA)

    # -------------------------
    # Nose
    # -------------------------
    cv2.line(
        frame,
        (cx, head_y - 5),
        (cx - 4, head_y + 20),
        (120, 135, 160),
        2,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        frame,
        (cx, head_y + 22),
        (10, 6),
        0,
        0,
        360,
        (120, 135, 160),
        -1,
        cv2.LINE_AA,
    )
    cv2.circle(frame, (cx - 5, head_y + 23), 2, (50, 50, 70), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + 5, head_y + 23), 2, (50, 50, 70), -1, cv2.LINE_AA)

    # -------------------------
    # Mouth (Dynamic talking)
    # -------------------------
    mouth_open = int(index / 12) % 2 == 0
    if mouth_open:
        ellipse(
            frame,
            (cx, head_y + 48),
            (18, 9),
            color=(80, 50, 150),
        )
        ellipse(
            frame,
            (cx, head_y + 48),
            (12, 5),
            color=(30, 20, 50),
        )
    else:
        cv2.line(
            frame,
            (cx - 16, head_y + 48),
            (cx + 16, head_y + 48),
            (80, 50, 150),
            4,
            cv2.LINE_AA,
        )

    # -------------------------
    # Static left arm
    # -------------------------
    shoulder_l = (cx - 65, 330)
    elbow_l = (cx - 115, 385)
    hand_l = (cx - 130, 420)

    cv2.line(frame, shoulder_l, elbow_l, (145, 165, 215), 25, cv2.LINE_AA)
    cv2.line(frame, elbow_l, hand_l, (145, 165, 215), 22, cv2.LINE_AA)
    ellipse(frame, hand_l, (18, 18), color=(145, 165, 215))

    # -------------------------
    # Moving right arm & Hand
    # -------------------------
    wave = math.sin(t * 3.0)
    shoulder_r = (cx + 65, 330)
    elbow_r = (cx + 120 + int(20 * wave), 360 + int(20 * wave))
    hand_r = (cx + 145 + int(70 * wave), 300 + int(55 * wave))

    cv2.line(frame, shoulder_r, elbow_r, (145, 165, 215), 25, cv2.LINE_AA)
    cv2.line(frame, elbow_r, hand_r, (145, 165, 215), 22, cv2.LINE_AA)
    ellipse(frame, hand_r, (20, 20), color=(145, 165, 215))

    # Fingers
    for finger_angle in (-0.8, -0.4, 0.0, 0.4, 0.8):
        dx = int(25 * math.cos(finger_angle))
        dy = int(25 * math.sin(finger_angle))
        cv2.line(
            frame,
            hand_r,
            (hand_r[0] + dx, hand_r[1] + dy),
            (145, 165, 215),
            7,
            cv2.LINE_AA,
        )


def generate(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output),
        fourcc,
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output}")

    try:
        for index in range(FRAMES):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[:] = (50, 40, 35)  # Dark neutral background in BGR

            # Simple floor
            cv2.line(frame, (0, 455), (WIDTH, 455), (100, 100, 100), 3)

            draw_person(frame, index)

            # Scene title
            cv2.putText(
                frame,
                "ANIMATED CREATOR VISION FIXTURE",
                (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            writer.write(frame)
    finally:
        writer.release()

    print(f"Created: {output}")
    print(f"Frames:  {FRAMES}")
    print(f"FPS:     {FPS}")
    print(f"Size:    {WIDTH}x{HEIGHT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "creator_test_video.mp4",
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
