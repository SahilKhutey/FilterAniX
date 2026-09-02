"""Multi-Track Final Video Compositor."""
import tempfile
from pathlib import Path
from typing import Optional

from src.media.ffmpeg import inspect_media, normalize_audio, mux_media


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
        """Executes full multi-track composition and loudness normalization."""
        video_p = Path(video_path).resolve()
        audio_src_p = Path(audio_source_path).resolve()
        output_p = Path(output_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)

        if not video_p.exists():
            raise FileNotFoundError(f"Input video file not found: {video_p}")
        if not audio_src_p.exists():
            raise FileNotFoundError(f"Audio source file not found: {audio_src_p}")

        # Check audio presence in source
        media_info = inspect_media(audio_src_p)
        temp_audio = None

        if media_info.has_audio:
            if normalize_loudness:
                temp_audio = output_p.parent / f"temp_norm_audio_{output_p.stem}.m4a"
                norm_ok = normalize_audio(
                    input_audio_source=audio_src_p,
                    output_audio_path=temp_audio,
                    target_lufs=self.target_lufs,
                    true_peak=self.true_peak,
                )
                if not norm_ok:
                    temp_audio = audio_src_p
            else:
                temp_audio = audio_src_p

        # Execute final multiplexing
        success = mux_media(
            video_path=video_p,
            audio_path=temp_audio,
            output_path=output_p,
            subtitles_path=subtitles_path,
            crf=self.crf,
            preset=self.preset,
        )

        # Cleanup temporary normalized audio file
        if temp_audio and temp_audio != audio_src_p and Path(temp_audio).exists():
            try:
                Path(temp_audio).unlink(missing_ok=True)
            except Exception:
                pass

        if not success or not output_p.exists():
            raise RuntimeError(f"Failed to generate composite video at: {output_p}")

        return str(output_p)
