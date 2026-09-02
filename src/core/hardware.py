"""Hardware and Environment Diagnostics."""
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.io.video_io import get_ffmpeg_executable, get_ffprobe_executable


@dataclass
class HardwareReport:
    """Comprehensive system hardware & capability report."""
    os_name: str
    python_version: str
    cpu_cores: int
    ffmpeg_available: bool
    ffmpeg_path: str
    ffprobe_available: bool
    cuda_available: bool
    gpu_name: str = "None (CPU Execution)"
    vram_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "OS": self.os_name,
            "Python": self.python_version,
            "CPU Cores": self.cpu_cores,
            "FFmpeg Available": self.ffmpeg_available,
            "FFmpeg Path": self.ffmpeg_path,
            "FFprobe Available": self.ffprobe_available,
            "CUDA Available": self.cuda_available,
            "GPU Device": self.gpu_name,
            "VRAM (GB)": round(self.vram_gb, 2),
        }


def get_hardware_report() -> HardwareReport:
    """Detects system hardware, Python, GPU, and FFmpeg environment."""
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    cpu_cores = os.cpu_count() or 4

    ffmpeg_path = "Not found"
    ffmpeg_avail = False
    try:
        ffmpeg_bin = get_ffmpeg_executable()
        if ffmpeg_bin:
            ffmpeg_path = ffmpeg_bin
            ffmpeg_avail = True
    except Exception:
        pass

    ffprobe_avail = get_ffprobe_executable() is not None

    cuda_avail = False
    gpu_name = "None (CPU Execution)"
    vram_gb = 0.0

    try:
        import torch
        if torch.cuda.is_available():
            cuda_avail = True
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass

    return HardwareReport(
        os_name=os_name,
        python_version=python_ver,
        cpu_cores=cpu_cores,
        ffmpeg_available=ffmpeg_avail,
        ffmpeg_path=ffmpeg_path,
        ffprobe_available=ffprobe_avail,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
    )
