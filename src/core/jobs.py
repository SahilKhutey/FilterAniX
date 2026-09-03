from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Event, Lock, Thread
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from src.core.events import EventBus, PipelineEvent, default_event_bus

TERMINAL_STATES = {
    "complete",
    "failed",
    "cancelled",
}


@dataclass
class Job:
    """Production job model with comprehensive live telemetry."""
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    current_frame: int = 0
    total_frames: int = 0
    fps: float = 0.0
    eta_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    stage: str = "queued"
    substage: str = ""
    stage_progress: float = 0.0
    message: str = ""
    current_output: Optional[str] = None
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

    @property
    def id(self) -> str:
        return self.job_id

    @property
    def frame(self) -> int:
        return self.current_frame

    @frame.setter
    def frame(self, val: int):
        self.current_frame = val

    @property
    def eta(self) -> float:
        return self.eta_seconds

    @eta.setter
    def eta(self, val: float):
        self.eta_seconds = val

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
            # Map aliases
            if "frame" in values and "current_frame" not in values:
                values["current_frame"] = values.pop("frame")
            if "eta" in values and "eta_seconds" not in values:
                values["eta_seconds"] = values.pop("eta")
            if "id" in values and "job_id" not in values:
                values["job_id"] = values.pop("id")

            for key, value in values.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "job_id": self.job_id,
                "id": self.job_id,
                "status": self.status,
                "stage": self.stage,
                "substage": self.substage,
                "progress": float(self.progress),
                "frame": int(self.current_frame),
                "current_frame": int(self.current_frame),
                "total_frames": int(self.total_frames),
                "fps": float(self.fps),
                "eta": float(self.eta_seconds),
                "eta_seconds": float(self.eta_seconds),
                "elapsed_seconds": float(self.elapsed_seconds),
                "stage_progress": float(self.stage_progress),
                "message": str(self.message),
                "current_output": self.current_output,
                "error": self.error,
                "result": self.result,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


class JobManager:
    """
    Thread-based production job manager with EventBus integration
    and real-time telemetry streaming.
    """

    def __init__(
        self,
        max_workers: int = 1,
        event_bus: Optional[EventBus] = None,
    ):
        self.max_workers = max(1, int(max_workers))
        self.jobs: Dict[str, Job] = {}
        self._queue: list[tuple[Job, Callable, tuple, dict]] = []
        self._lock = Lock()
        self._workers: list[Thread] = []
        self.event_bus = event_bus or default_event_bus

        # Subscribe to EventBus to receive real-time pipeline events
        self.event_bus.subscribe(self._on_pipeline_event)

        for index in range(self.max_workers):
            worker = Thread(
                target=self._worker_loop,
                name=f"filteranix-worker-{index}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _on_pipeline_event(self, event: PipelineEvent):
        """Updates matching job with incoming pipeline telemetry."""
        job = self.get(event.job_id)
        if job is not None and job.status not in TERMINAL_STATES:
            update_data: Dict[str, Any] = {
                "stage": event.stage,
                "progress": event.progress,
                "message": event.message,
            }
            if event.substage:
                update_data["substage"] = event.substage
            frame_val = getattr(event, "frame", 0) or getattr(event, "current_frame", 0)
            if frame_val > 0:
                update_data["current_frame"] = frame_val
            if event.total_frames > 0:
                update_data["total_frames"] = event.total_frames
            if event.fps > 0:
                update_data["fps"] = event.fps
            if event.eta_seconds is not None:
                update_data["eta_seconds"] = event.eta_seconds
            job.update(**update_data)

    def create(self) -> Job:
        job = Job(job_id=str(uuid4())[:8])
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
                    if isinstance(result, dict) and "final_video" in result:
                        job.current_output = result["final_video"]

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

    def snapshot(self, job_id: str) -> Optional[dict]:
        job = self.get(job_id)
        if job is None:
            return None
        return job.snapshot()

    def list_jobs(self) -> List[dict]:
        with self._lock:
            job_list = list(self.jobs.values())
        return [j.snapshot() for j in reversed(job_list)]

    def get_active_job(self) -> Optional[Job]:
        with self._lock:
            for job in reversed(list(self.jobs.values())):
                if job.status in ("running", "paused", "queued"):
                    return job
        return None
