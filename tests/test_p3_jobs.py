import time
from src.core.jobs import JobManager


def test_job_completes():
    manager = JobManager(max_workers=1)
    job = manager.create()

    def work(*args, **kwargs):
        time.sleep(0.05)
        return {"ok": True}

    manager.run_async(
        job,
        work,
    )

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if job.status == "complete":
            break
        time.sleep(0.01)

    assert job.status == "complete"
    assert job.result["ok"] is True
