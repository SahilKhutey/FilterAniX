from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemporalObservation:
    """
    Per-frame temporal information supplied by Phase 2 / motion analysis.

    optical_flow:
        Flow from the previous source frame to the current source frame.
        Shape: H x W x 2
        Channels:
            [..., 0] = horizontal displacement
            [..., 1] = vertical displacement

    scene_cut:
        True when the current frame begins a new visual scene.

    motion_magnitude:
        Optional precomputed motion magnitude.
        If omitted, it is calculated from optical_flow.
    """

    optical_flow: np.ndarray | None = None

    scene_cut: bool = False

    motion_magnitude: np.ndarray | None = None
