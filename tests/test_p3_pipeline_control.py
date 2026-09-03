import time
from src.core.jobs import JobManager


def test_pause_resume():
    manager = JobManager(max_workers=1)
    job = manager.create()

    def work(*args, **kwargs):
        current_job = kwargs.get("job")
        if current_job:
            current_job.update(message="working")
        time.sleep(0.2)
        return True

    manager.run_async(
        job,
        work,
    )

    time.sleep(0.03)
    manager.pause(job.job_id)
    assert job.status == "paused"

    manager.resume(job.job_id)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if job.status == "complete":
            break
        time.sleep(0.01)

    assert job.status == "complete"
