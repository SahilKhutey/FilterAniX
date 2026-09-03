from __future__ import annotations

from pathlib import Path
from typing import List

from src.core.project import Project


def cleanup_partial_files(root: str | Path) -> None:
    root = Path(root)
    for pattern in (
        "*.partial.mp4",
        "temp_*.mp4",
        "*.tmp",
    ):
        for path in root.rglob(pattern):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def recover_project(project: Project) -> List[str]:
    """Inspects manifest and project root on startup and resets crashed/in-flight stages for clean resumption."""
    manifest = project.load()
    recovery = []

    # Reset overall project status if crashed while running
    if manifest.get("status") == "running":
        manifest["status"] = "interrupted"
        manifest["error"] = "Application terminated before job completion."

    for name, stage in manifest.get("stages", {}).items():
        if stage.get("status") == "running":
            stage["status"] = "failed"
            stage["error"] = (
                "Previous process terminated while stage was running."
            )
            recovery.append(name)

    project.save(manifest)

    # Clean up stale locks and partial files
    lock_file = project.root / ".render.lock"
    lock_file.unlink(missing_ok=True)
    cleanup_partial_files(project.root)

    return recovery
