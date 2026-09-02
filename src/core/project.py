from __future__ import annotations

from pathlib import Path
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


STAGES = [
    "input",
    "vision",
    "artistic",
    "consistency",
    "lipsync",
    "composition",
    "validation",
    "export",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Project:
    """Manages project workspace directories and persistent stage manifest state."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root_dir = self.root
        self.manifest_path = self.root / "project.json"

        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

        for directory in [
            "source",
            "vision",
            "artistic",
            "consistency",
            "lipsync",
            "output",
            "reports",
            "checkpoints",
        ]:
            (self.root / directory).mkdir(
                exist_ok=True
            )

    def create(self, name: str) -> Dict[str, Any]:
        manifest = {
            "project_id": str(uuid.uuid4()),
            "name": name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "created",
            "stages": {
                stage: {
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "error": None,
                    "output": None,
                }
                for stage in STAGES
            },
        }

        self.save(manifest)
        return manifest

    def load(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            alt = self.root / "manifest.json"
            if alt.exists():
                with open(alt, "r", encoding="utf-8") as f:
                    return json.load(f)
            raise FileNotFoundError(self.manifest_path)

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def save(self, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = utc_now()
        temp = self.manifest_path.with_suffix(".tmp")

        with open(
            temp,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                manifest,
                f,
                indent=2,
            )

        temp.replace(self.manifest_path)

    def update_stage(
        self,
        stage: str,
        status: str,
        output: Any = None,
        error: Any = None,
    ) -> None:
        manifest = self.load()
        if "stages" not in manifest or stage not in manifest["stages"]:
            if "stages" not in manifest:
                manifest["stages"] = {}
            manifest["stages"][stage] = {
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "error": None,
                "output": None,
            }

        state = manifest["stages"][stage]
        state["status"] = status

        if status == "running":
            state["started_at"] = utc_now()

        if status in (
            "complete",
            "failed",
            "skipped",
        ):
            state["finished_at"] = utc_now()

        if output:
            state["output"] = str(output)

        if error:
            state["error"] = str(error)

        self.save(manifest)

    def stage_complete(self, stage: str) -> bool:
        try:
            manifest = self.load()
            return manifest.get("stages", {}).get(stage, {}).get("status") == "complete"
        except Exception:
            return False
