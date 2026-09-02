from pathlib import Path
import json
import cv2

from .engine import VisionEngine


class VideoAnalyzer:
    def __init__(self, vision_engine: VisionEngine):
        self.engine = vision_engine

    def analyze(
        self,
        input_path: Path,
        output_dir: Path,
        save_masks: bool = False,
        make_annotated_video: bool = True,
        progress_callback=None,
    ):
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "frames"
        masks_dir = output_dir / "masks"

        if save_masks:
            masks_dir.mkdir(exist_ok=True)

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = None
        if make_annotated_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_dir / "annotated.mp4"),
                fourcc,
                fps,
                (width, height),
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Could not create annotated output.")

        jsonl_path = output_dir / "vision.jsonl"
        records = 0
        face_frames = 0
        pose_frames = 0
        hand_frames = 0

        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    timestamp = records / fps
                    vision = self.engine.process_frame(
                        frame,
                        records,
                        timestamp,
                    )

                    jsonl.write(
                        json.dumps(
                            vision.to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )

                    if vision.faces:
                        face_frames += 1
                    if vision.pose:
                        pose_frames += 1
                    if vision.hands:
                        hand_frames += 1

                    if save_masks and vision.person_mask is not None:
                        # Re-run the lightweight pose segmentation mask is
                        # intentionally not serialized in JSON. This option
                        # can be expanded in Phase 3 when masks become assets.
                        pass

                    if writer is not None:
                        writer.write(
                            self.engine.draw_overlay(frame, vision)
                        )

                    records += 1

                    if progress_callback and total:
                        progress_callback(int(records * 100 / total))

            finally:
                cap.release()
                if writer is not None:
                    writer.release()

        summary = {
            "input": str(input_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frames": records,
            "duration": records / fps if fps else 0,
            "frames_with_face": face_frames,
            "frames_with_pose": pose_frames,
            "frames_with_hands": hand_frames,
            "outputs": {
                "vision_jsonl": str(jsonl_path),
                "annotated_video": str(output_dir / "annotated.mp4")
                if make_annotated_video else None,
            },
        }

        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        return summary


def analyze_video(input_path: str | Path, output_path: str | Path) -> dict:
    """Convenience function to analyze video and write vision.jsonl."""
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_dir = output_p.parent if output_p.suffix == ".jsonl" else output_p
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = VisionEngine()
    analyzer = VideoAnalyzer(engine)
    res = analyzer.analyze(input_p, output_dir)
    return res

