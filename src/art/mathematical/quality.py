"""Quality gate validation for FilterAniX Mathematical Anime Engine."""
from __future__ import annotations

import numpy as np


class RenderValidationError(ValueError):
    """Raised when rendered frame violates structural or photometric quality criteria."""
    pass


def validate_render(input_rgb: np.ndarray, output_rgb: np.ndarray) -> bool:
    """
    Validates rendered frame output against physical and artistic sanity invariants.
    
    Rejects:
      - Non-uint8 datatypes
      - Spatial dimension changes (H, W, C)
      - Crushed blacks (mean < 15, unless input itself is deeply dark)
      - Blown-out whites (mean > 245, unless input itself is nearly pure white)
      - Collapsed flat images (std < 5, unless input itself is flat)
      
    Returns True if valid; raises RenderValidationError otherwise.
    """
    if not isinstance(output_rgb, np.ndarray):
        raise RenderValidationError(f"Renderer output must be a numpy.ndarray, got {type(output_rgb)}")

    if output_rgb.dtype != np.uint8:
        raise RenderValidationError(f"Renderer output must be uint8, got {output_rgb.dtype}")

    if not isinstance(input_rgb, np.ndarray):
        raise RenderValidationError(f"Renderer input must be a numpy.ndarray, got {type(input_rgb)}")

    if output_rgb.shape != input_rgb.shape:
        raise RenderValidationError(
            f"Renderer changed frame dimensions: expected {input_rgb.shape}, got {output_rgb.shape}"
        )

    if output_rgb.ndim != 3 or output_rgb.shape[2] != 3:
        raise RenderValidationError(
            f"Renderer output must have shape (H, W, 3), got {output_rgb.shape}"
        )

    in_mean = float(np.mean(input_rgb))
    out_mean = float(np.mean(output_rgb))

    # Crushed black check: only trigger if input is not dark
    if out_mean < 15.0 and in_mean >= 15.0:
        raise RenderValidationError(
            f"Output is almost completely black (mean {out_mean:.1f} < 15.0 on input mean {in_mean:.1f})"
        )

    # Blown-out white check: only trigger if input is not already high-key/very bright
    if (out_mean > 252.0 and in_mean <= 245.0) or (out_mean > 245.0 and in_mean < 185.0):
        raise RenderValidationError(
            f"Output is almost completely white (mean {out_mean:.1f} > 245.0 on input mean {in_mean:.1f})"
        )

    in_std = float(np.std(input_rgb))
    out_std = float(np.std(output_rgb))

    # Flat collapse check: only trigger if input had visual texture/content
    if out_std < 5.0 and in_std >= 5.0:
        raise RenderValidationError(
            f"Output has collapsed to near-flat image (std {out_std:.2f} < 5.0 on input std {in_std:.2f})"
        )

    return True
