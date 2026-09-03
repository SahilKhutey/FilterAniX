from __future__ import annotations

from pathlib import Path
from typing import Optional

from .ffmpeg import run_ffmpeg, inspect_media, normalize_audio, mux_media


def compose_final_video(
    animated_video: str | Path,
    audio_source: str | Path,
    output: str | Path,
    subtitles: str | None = None,
    crf: int = 18,
    preset: str = "medium",
    audio_bitrate: str = "192k",
) -> Path:
    """Composites animated video frames with the original voice audio stream and applies EBU R128 loudness normalization."""
    output_path = Path(output).resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_details = inspect_media(audio_source)

    if source_details.has_audio:
        # Standard composition with EBU R128 loudness normalization
        args = [
            "-i",
            str(animated_video),
            "-i",
            str(audio_source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            audio_bitrate,
            "-ar",
            "48000",
            "-ac",
            "2",
            "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-movflags",
            "+faststart",
            "-shortest",
        ]

        if subtitles and Path(subtitles).exists():
            args.extend(
                [
                    "-vf",
                    f"subtitles={subtitles}",
                ]
            )

        args.append(str(output_path))
        result = run_ffmpeg(args, check=False)

        if result.returncode != 0:
            # Fallback attempt: audio mux without loudnorm (e.g. if audio was too short for 3-second loudnorm window)
            fallback_args = [
                "-i",
                str(animated_video),
                "-i",
                str(audio_source),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                audio_bitrate,
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ]
            fallback_res = run_ffmpeg(fallback_args, check=False)
            if fallback_res.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg audio composition failed on audio-enabled source:\n"
                    f"Loudnorm attempt error: {result.stderr}\n"
                    f"Fallback attempt error: {fallback_res.stderr}"
                )
    else:
        # Video-only input (no audio track in source)
        args = [
            "-i",
            str(animated_video),
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        result = run_ffmpeg(args, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg video composition failed: {result.stderr}")

    return output_path


class VideoCompositor:
    """Composites stylized animated frames with normalized voice audio and optional subtitles."""

    def __init__(
        self,
        target_lufs: float = -14.0,
        true_peak: float = -1.5,
        crf: int = 18,
        preset: str = "medium",
    ):
        self.target_lufs = target_lufs
        self.true_peak = true_peak
        self.crf = crf
        self.preset = preset

    def compose(
        self,
        video_path: str | Path,
        audio_source_path: str | Path,
        output_path: str | Path,
        subtitles_path: Optional[str | Path] = None,
        normalize_loudness: bool = True,
    ) -> str:
        res = compose_final_video(
            animated_video=video_path,
            audio_source=audio_source_path,
            output=output_path,
            subtitles=subtitles_path,
            crf=self.crf,
            preset=self.preset,
        )
        return str(res)
