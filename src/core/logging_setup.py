from __future__ import annotations

import logging
from pathlib import Path


def setup_logging() -> logging.Logger:
    """Configures project-wide structured logging to logs/animated_creator.log and stderr."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("logs/animated_creator.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("animated_creator")
