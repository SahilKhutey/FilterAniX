from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def get_stage_logger(stage_name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Returns a stage-specific logger that writes to logs/<stage_name>.log and stream."""
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"filteranix.{stage_name}")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler(path / f"{stage_name}.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(stream_handler)

    return logger


def setup_all_logging(log_dir: str | Path = "logs") -> dict[str, logging.Logger]:
    """Configures standard loggers for FilterAniX stages."""
    stages = ["pipeline", "vision", "render", "audio", "export"]
    return {stage: get_stage_logger(stage, log_dir) for stage in stages}


setup_logging = lambda: get_stage_logger("pipeline")
