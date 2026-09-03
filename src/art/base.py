"""Base abstractions for FilterAniX artistic renderers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class ArtisticRenderer(ABC):
    """
    Abstract base class for all FilterAniX artistic renderers.
    Standardizes lifecycle and frame transformation interface.
    """

    @abstractmethod
    def reset(self) -> None:
        """Resets temporal memory and internal frame buffers (e.g. on scene cuts)."""
        pass

    @abstractmethod
    def render(
        self,
        rgb: np.ndarray,
        vision_data: Optional[Any] = None,
        scene_cut: bool = False,
        stabilize: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Executes artistic transformation on a single RGB frame (uint8, HxWx3).
        Returns transformed RGB frame with identical spatial dimensions and uint8 dtype.
        """
        pass
