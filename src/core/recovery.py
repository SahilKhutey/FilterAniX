from __future__ import annotations

from typing import List
from src.core.project import Project


def recover_project(project: Project) -> List[str]:
    """Inspects manifest on startup and resets crashed/in-flight stages to failed for clean resumption."""
    manifest = project.load()
    recovery = []

    for name, stage in manifest.get("stages", {}).items():
        if stage.get("status") == "running":
            stage["status"] = "failed"
            stage["error"] = (
                "Previous process terminated while stage was running."
            )
            recovery.append(name)

    project.save(manifest)
    return recovery
