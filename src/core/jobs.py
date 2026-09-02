from __future__ import annotations

from dataclasses import dataclass
from threading import Thread
from uuid import uuid4
from typing import Any, Callable, Dict, Optional


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    result: Any = None


class JobManager:
    """Manages asynchronous pipeline execution jobs for non-blocking UI and API operations."""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}

    def create(self) -> Job:
        job = Job(job_id=str(uuid4()))
        self.jobs[job.job_id] = job
        return job

    def run_async(
        self,
        job: Job,
        function: Callable,
        *args,
        **kwargs,
    ) -> Job:
        def worker():
            job.status = "running"
            try:
                res = function(*args, **kwargs)
                job.result = res
                job.progress = 1.0
                job.status = "complete"
            except Exception as exc:
                job.error = str(exc)
                job.status = "failed"

        thread = Thread(
            target=worker,
            daemon=True,
        )
        thread.start()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)
