"""Tests for upgraded Job telemetry, EventBus event propagation, and snapshots."""
import time
import pytest
from src.core.jobs import Job, JobManager
from src.core.events import EventBus, PipelineEvent


def test_job_creation():
    manager = JobManager(max_workers=1)
    job = manager.create()
    assert job.job_id
    assert job.id == job.job_id
    assert job.status == "queued"
    assert job.frame == 0
    assert job.total_frames == 0
    assert job.fps == 0.0
    assert job.eta == 0.0


def test_job_progress_and_aliases():
    job = Job(job_id="test-123")
    job.update(
        stage="artistic",
        substage="MTH-09 Lighting Field",
        progress=0.63,
        frame=151,
        total_frames=240,
        fps=21.4,
        eta=4.2,
        message="Applying mathematical lighting field",
    )
    assert job.stage == "artistic"
    assert job.substage == "MTH-09 Lighting Field"
    assert job.progress == 0.63
    assert job.current_frame == 151
    assert job.frame == 151
    assert job.total_frames == 240
    assert job.fps == 21.4
    assert job.eta_seconds == 4.2
    assert job.eta == 4.2
    assert job.message == "Applying mathematical lighting field"


def test_job_snapshot_schema():
    job = Job(job_id="test-snap")
    job.update(
        stage="mth09_lighting",
        progress=0.63,
        frame=151,
        total_frames=240,
        fps=21.4,
        eta=4.2,
        message="Applying mathematical lighting field",
    )
    snap = job.snapshot()
    assert snap["job_id"] == "test-snap"
    assert snap["id"] == "test-snap"
    assert snap["stage"] == "mth09_lighting"
    assert snap["progress"] == 0.63
    assert snap["frame"] == 151
    assert snap["current_frame"] == 151
    assert snap["total_frames"] == 240
    assert snap["fps"] == 21.4
    assert snap["eta"] == 4.2
    assert snap["eta_seconds"] == 4.2
    assert snap["message"] == "Applying mathematical lighting field"


def test_job_completion():
    manager = JobManager(max_workers=1)
    job = manager.create()

    def work():
        return {"final_video": "master.mp4"}

    manager.run_async(job, work)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if job.status == "complete":
            break
        time.sleep(0.01)

    assert job.status == "complete"
    assert job.progress == 1.0
    assert job.result == {"final_video": "master.mp4"}
    assert job.current_output == "master.mp4"


def test_job_failure():
    manager = JobManager(max_workers=1)
    job = manager.create()

    def failing_work():
        raise ValueError("Corrupted frame data")

    manager.run_async(job, failing_work)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if job.status == "failed":
            break
        time.sleep(0.01)

    assert job.status == "failed"
    assert "Corrupted frame data" in str(job.error)


def test_eventbus_updates_job():
    event_bus = EventBus()
    manager = JobManager(max_workers=1, event_bus=event_bus)
    job = manager.create()

    # Emit pipeline event
    event = PipelineEvent(
        job_id=job.job_id,
        stage="artistic",
        substage="MTH-05 Edge Field",
        progress=0.45,
        frame=100,
        total_frames=200,
        fps=24.5,
        eta_seconds=4.0,
        message="Extracting multi-scale gradient edges",
    )
    event_bus.emit(event)

    assert job.stage == "artistic"
    assert job.substage == "MTH-05 Edge Field"
    assert job.progress == 0.45
    assert job.current_frame == 100
    assert job.total_frames == 200
    assert job.fps == 24.5
    assert job.eta_seconds == 4.0
