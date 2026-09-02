"""Hardware and Environment Diagnostics."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.media.ffmpeg import get_ffmpeg_executable, get_ffprobe_executable


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def get_gpu() -> Optional[str]:
    if not command_exists("nvidia-smi"):
        # Check torch fallback
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_name(0)
        except Exception:
            pass
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def select_live_backend(hardware: Dict[str, Any]) -> str:
    """Safety switch: never automatically launch expensive diffusion for real-time live webcam."""
    if hardware.get("gpu"):
        return "fast_gpu"
    return "opencv"


def system_info() -> Dict[str, Any]:
    ffmpeg_ok = False
    try:
        ffmpeg_ok = get_ffmpeg_executable() is not None
    except Exception:
        pass

    ffprobe_ok = False
    try:
        ffprobe_ok = get_ffprobe_executable() is not None
    except Exception:
        pass

    return {
        "os": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "gpu": get_gpu(),
    }


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

    gpu_str = get_gpu()
    if gpu_str:
        cuda_avail = True
        gpu_name = gpu_str

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


if __name__ == "__main__":
    import json
    print(json.dumps(system_info(), indent=2))
