"""Project Manager and Manifest State Machine."""
import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ProjectStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    PHASE_1 = "PHASE_1_INPUT"
    PHASE_2 = "PHASE_2_VISION"
    PHASE_3 = "PHASE_3_STYLE"
    PHASE_4 = "PHASE_4_CONSISTENCY"
    PHASE_5 = "PHASE_5_COMPOSITION"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class ProjectManifest:
    """Project Source of Truth tracking stages, outputs, style, and status."""
    project_id: str
    project_version: str = "1.0"
    status: str = ProjectStatus.CREATED.value
    style_key: str = "anime_creator"
    input_video_path: str = ""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    updated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_version": self.project_version,
            "status": self.status,
            "style_key": self.style_key,
            "input_video_path": self.input_video_path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stages": self.stages,
            "outputs": self.outputs,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectManifest":
        return cls(
            project_id=d.get("project_id", "project"),
            project_version=d.get("project_version", "1.0"),
            status=d.get("status", ProjectStatus.CREATED.value),
            style_key=d.get("style_key", "anime_creator"),
            input_video_path=d.get("input_video_path", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            stages=d.get("stages", {}),
            outputs=d.get("outputs", {}),
            errors=d.get("errors", []),
        )


class Project:
    """Manages project workspace directories and manifest state machine."""

    def __init__(self, project_dir: str | Path):
        self.root_dir = Path(project_dir).resolve()
        self.input_dir = self.root_dir / "input"
        self.phase1_dir = self.root_dir / "phase1"
        self.phase2_dir = self.root_dir / "phase2"
        self.phase3_dir = self.root_dir / "phase3"
        self.phase4_dir = self.root_dir / "phase4"
        self.phase5_dir = self.root_dir / "phase5"
        self.export_dir = self.root_dir / "export"
        self.logs_dir = self.root_dir / "logs"
        self.manifest_path = self.root_dir / "manifest.json"
        self.log_file = self.logs_dir / "pipeline.log"

        self._init_directories()
        self.manifest = self._load_or_create_manifest()

    def _init_directories(self):
        for p in [
            self.input_dir,
            self.phase1_dir,
            self.phase2_dir,
            self.phase3_dir,
            self.phase4_dir,
            self.phase5_dir,
            self.export_dir,
            self.logs_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def _load_or_create_manifest(self) -> ProjectManifest:
        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return ProjectManifest.from_dict(json.load(f))
        
        manifest = ProjectManifest(project_id=self.root_dir.name)
        self.save_manifest(manifest)
        return manifest

    def save_manifest(self, manifest: Optional[ProjectManifest] = None):
        if manifest:
            self.manifest = manifest
        self.manifest.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.manifest.to_dict(), f, indent=2)

    def set_status(self, status: ProjectStatus):
        self.manifest.status = status.value
        self.save_manifest()

    def record_stage_completed(self, stage_name: str, details: Dict[str, Any]):
        self.manifest.stages[stage_name] = {
            "status": "COMPLETED",
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "details": details,
        }
        self.save_manifest()

    def record_error(self, error_msg: str):
        self.manifest.status = ProjectStatus.FAILED.value
        self.manifest.errors.append(error_msg)
        self.save_manifest()
