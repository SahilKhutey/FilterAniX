"""Animated Creator — Phase 1 Application Entry Point."""
import argparse
import sys
from pathlib import Path

# Ensure root directory in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.processing.pipeline import VideoPipeline, FrameProcessor
from src.core.models import VideoMetadata
from src.io.video_io import inspect_video


def run_cli(input_path: str, output_path: str):
    """Executes the Phase 1 video pipeline in headless CLI mode."""
    print("==================================================")
    print("   ANIMATED CREATOR — PHASE 1 CLI RUNNER          ")
    print("==================================================")
    
    input_p = Path(input_path)
    output_p = Path(output_path)
    
    if not input_p.exists():
        print(f"[ERROR] Input video does not exist: {input_p}")
        sys.exit(1)

    print(f"[*] Inspecting video metadata: {input_p}...")
    metadata = inspect_video(input_p)
    print(f"[+] Metadata: {metadata.resolution_str} @ {metadata.fps:.2f} FPS | Frames: {metadata.frame_count} | Audio: {'Yes' if metadata.has_audio else 'No'}")

    pipeline = VideoPipeline(frame_processor=FrameProcessor())

    def on_progress(p):
        print(f"\r[*] {p.status_message} (Speed: {p.fps:.1f} FPS, ETA: {p.eta_sec:.1f}s)", end="", flush=True)

    print(f"[*] Executing pipeline: {input_p} -> {output_p}...")
    final_out = pipeline.process_video(
        input_path=input_p,
        output_path=output_p,
        progress_callback=on_progress,
    )
    print(f"\n[SUCCESS] Completed! Final MP4 with preserved audio written to: {final_out}")


def run_gui():
    """Launches the PyQt6 Desktop GUI."""
    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def main():
    parser = argparse.ArgumentParser(description="Animated Creator — Phase 1: Foundation & Video Pipeline")
    parser.add_argument("--input", "-i", type=str, default=None, help="Input video path for CLI mode")
    parser.add_argument("--output", "-o", type=str, default="samples/output_phase1.mp4", help="Output video path")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI mode")

    args = parser.parse_args()

    if args.cli or args.input is not None:
        if not args.input:
            print("[ERROR] Please provide --input <file.mp4> when running in CLI mode.")
            sys.exit(1)
        run_cli(args.input, args.output)
    else:
        run_gui()


if __name__ == "__main__":
    main()
