from __future__ import annotations

import shutil
from typing import Any

import psutil


class ResourceMonitor:

    @staticmethod
    def snapshot(path: str = ".") -> dict[str, Any]:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = shutil.disk_usage(path)

        result = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count(),
            },
            "memory": {
                "total_gb": round(
                    memory.total / 1024**3,
                    2,
                ),
                "used_gb": round(
                    memory.used / 1024**3,
                    2,
                ),
                "available_gb": round(
                    memory.available / 1024**3,
                    2,
                ),
                "percent": memory.percent,
            },
            "swap": {
                "total_gb": round(
                    swap.total / 1024**3,
                    2,
                ),
                "used_gb": round(
                    swap.used / 1024**3,
                    2,
                ),
                "percent": swap.percent,
            },
            "disk": {
                "total_gb": round(
                    disk.total / 1024**3,
                    2,
                ),
                "free_gb": round(
                    disk.free / 1024**3,
                    2,
                ),
                "used_percent": round(
                    (disk.used / disk.total) * 100,
                    2,
                ),
            },
        }

        try:
            import torch

            result["cuda"] = {
                "available": bool(
                    torch.cuda.is_available()
                )
            }

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                result["cuda"].update({
                    "device": torch.cuda.get_device_name(device),
                    "allocated_gb": round(
                        torch.cuda.memory_allocated(device) / 1024**3,
                        2,
                    ),
                    "reserved_gb": round(
                        torch.cuda.memory_reserved(device) / 1024**3,
                        2,
                    ),
                })
        except Exception as exc:
            result["cuda"] = {
                "available": False,
                "error": str(exc),
            }

        return result

    @staticmethod
    def healthy(
        snapshot: dict,
        minimum_free_disk_gb: float = 1.0,
    ) -> tuple[bool, list[str]]:
        errors = []

        disk = snapshot.get("disk", {})
        if disk.get("free_gb", 0) < minimum_free_disk_gb:
            errors.append(
                f"Low disk space: "
                f"{disk.get('free_gb', 0):.2f} GB (minimum required: {minimum_free_disk_gb:.2f} GB)"
            )

        memory = snapshot.get("memory", {})
        if memory.get("percent", 0) >= 98:
            errors.append("System memory critically high (>= 98%).")

        return len(errors) == 0, errors
