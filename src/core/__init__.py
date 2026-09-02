"""Core Architecture Package."""
from src.core.state import StageState, StageStatus
from src.core.project import Project, STAGES, utc_now
from src.core.pipeline import PipelineManager
from src.core.jobs import Job, JobManager
from src.core.recovery import recover_project
from src.core.hardware import (
    command_exists,
    get_gpu,
    system_info,
    select_live_backend,
    get_hardware_report,
    HardwareReport,
)
from src.core.config import load_config, load_json, load_styles
from src.core.logging_setup import setup_logging

__all__ = [
    "StageState",
    "StageStatus",
    "Project",
    "STAGES",
    "utc_now",
    "PipelineManager",
    "Job",
    "JobManager",
    "recover_project",
    "command_exists",
    "get_gpu",
    "system_info",
    "select_live_backend",
    "get_hardware_report",
    "HardwareReport",
    "load_config",
    "load_json",
    "load_styles",
    "setup_logging",
]
