"""Automated Tests for Phase 6 Project Management and Manifest State Machine."""
from pathlib import Path
import pytest

from src.core.project import Project, ProjectStatus, ProjectManifest


def test_project_workspace_and_manifest(tmp_path):
    """Verifies that Project properly sets up the workspace and persists manifest state transitions."""
    proj_dir = tmp_path / "test_creator_proj"
    project = Project(proj_dir)

    # Check directory hierarchy
    assert (proj_dir / "input").exists()
    assert (proj_dir / "phase1").exists()
    assert (proj_dir / "phase2").exists()
    assert (proj_dir / "phase3").exists()
    assert (proj_dir / "phase4").exists()
    assert (proj_dir / "phase5").exists()
    assert (proj_dir / "export").exists()
    assert (proj_dir / "logs").exists()
    assert (proj_dir / "manifest.json").exists()

    # State transitions
    assert project.manifest.status == ProjectStatus.CREATED.value
    project.set_status(ProjectStatus.PHASE_1)
    assert project.manifest.status == ProjectStatus.PHASE_1.value

    # Reload from disk
    reloaded_project = Project(proj_dir)
    assert reloaded_project.manifest.status == ProjectStatus.PHASE_1.value
    assert reloaded_project.manifest.project_id == "test_creator_proj"
