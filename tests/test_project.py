from pathlib import Path
import pytest

from src.core.project import Project, STAGES
from src.core.recovery import recover_project
from src.core.jobs import JobManager
from src.core.config import load_config


def test_project_creation_and_manifest(tmp_path):
    proj_dir = tmp_path / "test_creator_proj"
    project = Project(proj_dir)

    # Check directory structure
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
        assert (proj_dir / directory).exists()

    manifest = project.create("test_creator_proj")
    assert manifest["name"] == "test_creator_proj"
    assert manifest["status"] == "created"
    assert all(stage in manifest["stages"] for stage in STAGES)

    # Update stage state
    project.update_stage("input", "running")
    assert project.load()["stages"]["input"]["status"] == "running"

    project.update_stage("input", "complete", output=str(proj_dir / "source" / "test.mp4"))
    assert project.stage_complete("input") is True
    assert project.load()["stages"]["input"]["output"] == str(proj_dir / "source" / "test.mp4")


def test_project_recovery(tmp_path):
    proj_dir = tmp_path / "test_recovery_proj"
    project = Project(proj_dir)
    project.create("recovery_test")

    # Simulate crash mid-stage
    project.update_stage("vision", "running")

    recovered = recover_project(project)
    assert "vision" in recovered
    assert project.load()["stages"]["vision"]["status"] == "failed"


def test_job_manager_async_execution():
    manager = JobManager()
    job = manager.create()
    assert job.status == "queued"

    def dummy_task(x):
        return x * 2

    manager.run_async(job, dummy_task, 21)

    import time
    for _ in range(50):
        if job.status in {"complete", "failed"}:
            break
        time.sleep(0.02)

    assert job.status == "complete"
    assert job.result == 42


def test_config_loader():
    cfg = load_config("configs/default.yaml")
    assert cfg.get("name") == "balanced"
    assert "art" in cfg
