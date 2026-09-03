import time
from src.core.cancellation import JobControl
from src.core.jobs import JobManager


def test_job_cancel():
    manager = JobManager(max_workers=1)
    job = manager.create()

    def work(*args, **kwargs):
        control = JobControl(kwargs.get("job"))
        for _ in range(100):
            control.check()
            time.sleep(0.01)
        return True

    manager.run_async(
        job,
        work,
    )

    time.sleep(0.05)
    manager.cancel(job.job_id)

    deadline = time.time() + 2.0
    while time.time() < deadline:
        if job.status == "cancelled":
            break
        time.sleep(0.01)

    assert job.status == "cancelled"
