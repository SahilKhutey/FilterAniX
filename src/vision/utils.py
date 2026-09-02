from __future__ import annotations


def clamp_confidence(
    value: float | None,
) -> float:
    """Clamps a detector confidence value between 0.0 and 1.0, gracefully handling invalid/None values."""
    if value is None:
        return 0.0

    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(1.0, value),
    )
