from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

TERMINAL_STATES = {
    "complete",
    "failed",
    "cancelled",
}


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    stage: str = "queued"
    stage_progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    result: Any = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    _cancel_event: Event = field(
        default_factory=Event,
        repr=False,
    )
    _pause_event: Event = field(
        default_factory=Event,
        repr=False,
    )
    _lock: Lock = field(
        default_factory=Lock,
        repr=False,
    )

    def __post_init__(self):
        # Set means "not paused".
        self._pause_event.set()

    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def pause(self):
        if self.status in ("running", "queued"):
            self._pause_event.clear()
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self._pause_event.set()
            self.status = "running"

    def wait_if_paused(self):
        self._pause_event.wait()

    def update(self, **values):
        with self._lock:
            for key, value in values.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "progress": self.progress,
                "current_frame": self.current_frame,
                "total_frames": self.total_frames,
                "fps": self.fps,
                "eta_seconds": self.eta_seconds,
                "elapsed_seconds": self.elapsed_seconds,
                "stage": self.stage,
                "stage_progress": self.stage_progress,
                "message": self.message,
                "error": self.error,
                "result": self.result,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class JobManager:
    """
    Thread-based production job manager.
    The pipeline itself periodically calls the supplied cancellation/pause checks.
    """

    def __init__(self, max_workers: int = 1):
        self.max_workers = max(1, int(max_workers))
        self.jobs: Dict[str, Job] = {}
        self._queue: list[tuple[Job, Callable, tuple, dict]] = []
        self._lock = Lock()
        self._workers: list[Thread] = []

        for index in range(self.max_workers):
            worker = Thread(
                target=self._worker_loop,
                name=f"filteranix-worker-{index}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def create(self) -> Job:
        job = Job(job_id=str(uuid4()))
        with self._lock:
            self.jobs[job.job_id] = job
        return job

    def run_async(
        self,
        job: Job,
        function: Callable,
        *args,
        **kwargs,
    ) -> Job:
        with self._lock:
            self._queue.append(
                (job, function, args, kwargs)
            )
        return job

    def _worker_loop(self):
        while True:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)

            if item is None:
                import time

                time.sleep(0.05)
                continue

            job, function, args, kwargs = item

            if job.is_cancelled():
                job.status = "cancelled"
                continue

            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()

            try:
                # If function accepts 'job' keyword arg, pass it, otherwise call directly
                import inspect
                sig = inspect.signature(function)
                if "job" in sig.parameters:
                    kwargs["job"] = job
                result = function(
                    *args,
                    **kwargs,
                )

                if job.is_cancelled():
                    job.status = "cancelled"
                else:
                    job.result = result
                    job.progress = 1.0
                    job.status = "complete"

            except Exception as exc:
                job.error = str(exc)

                if job.is_cancelled() or "cancelled" in str(exc).lower():
                    job.status = "cancelled"
                else:
                    job.status = "failed"

            finally:
                job.finished_at = datetime.now(timezone.utc).isoformat()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)

        if job is None:
            return False

        job.cancel()

        if job.status == "queued":
            job.status = "cancelled"

        return True

    def pause(self, job_id: str) -> bool:
        job = self.get(job_id)

        if job is None:
            return False

        job.pause()
        return True

    def resume(self, job_id: str) -> bool:
        job = self.get(job_id)

        if job is None:
            return False

        job.resume()
        return True

    def status(self, job_id: str) -> dict:
        job = self.get(job_id)

        if job is None:
            return {
                "status": "not_found"
            }

        return job.snapshot()
