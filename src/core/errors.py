from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class FilterAniXError(Exception):
    """Base FilterAniX exception."""


class InputError(FilterAniXError):
    """Invalid or unreadable input."""


class VisionError(FilterAniXError):
    """Vision processing failure."""


class RenderingError(FilterAniXError):
    """Mathematical rendering failure."""


class TemporalError(FilterAniXError):
    """Temporal consistency failure."""


class AudioError(FilterAniXError):
    """Audio processing failure."""


class CompositionError(FilterAniXError):
    """Final A/V composition failure."""


class ExportError(FilterAniXError):
    """Export failure."""


class ValidationError(FilterAniXError):
    """Output validation failure."""


def write_error_manifest(
    output_path: str | Path,
    stage: str,
    error: Exception | str,
    frame_index: Optional[int] = None,
    input_path: Optional[str | Path] = None,
    details: Optional[dict[str, Any]] = None,
) -> Path:
    """Writes a structured error manifest to disk."""
    out = Path(output_path)
    if out.is_dir() or not out.suffix:
        manifest_path = out / "error.json"
    else:
        manifest_path = out

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "status": "failed",
        "stage": stage,
        "frame_index": frame_index,
        "error_type": type(error).__name__ if isinstance(error, Exception) else "Error",
        "message": str(error),
        "input": str(input_path) if input_path else None,
        "details": details or {},
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return manifest_path
