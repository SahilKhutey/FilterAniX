"""Multi-Stage Project Logging Setup."""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_project_logger(log_file_path: Optional[str | Path] = None) -> logging.Logger:
    """Configures project logger outputting to both console and project log file."""
    logger = logging.getLogger("FilterAniX")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    if log_file_path:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)

    return logger
