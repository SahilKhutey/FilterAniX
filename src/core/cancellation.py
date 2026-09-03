from __future__ import annotations

from dataclasses import dataclass


class JobCancelledError(RuntimeError):
    pass


@dataclass
class JobControl:
    job: object | None = None

    def check(self):
        if self.job is None:
            return

        if hasattr(self.job, "is_cancelled") and self.job.is_cancelled():
            raise JobCancelledError(
                "Render cancelled by user."
            )

        if hasattr(self.job, "wait_if_paused"):
            self.job.wait_if_paused()

        if hasattr(self.job, "is_cancelled") and self.job.is_cancelled():
            raise JobCancelledError(
                "Render cancelled while paused."
            )

    def update(
        self,
        *,
        stage: str | None = None,
        substage: str | None = None,
        progress: float | None = None,
        current_frame: int | None = None,
        total_frames: int | None = None,
        fps: float | None = None,
        eta_seconds: float | None = None,
        elapsed_seconds: float | None = None,
        message: str | None = None,
        current_output: str | None = None,
    ):
        if self.job is None or not hasattr(self.job, "update"):
            return

        values = {}

        if stage is not None:
            values["stage"] = stage

        if substage is not None:
            values["substage"] = substage

        if progress is not None:
            values["progress"] = max(
                0.0,
                min(1.0, float(progress)),
            )

        if current_frame is not None:
            values["current_frame"] = int(current_frame)

        if total_frames is not None:
            values["total_frames"] = int(total_frames)

        if fps is not None:
            values["fps"] = float(fps)

        if eta_seconds is not None:
            values["eta_seconds"] = max(
                0.0,
                float(eta_seconds),
            )

        if elapsed_seconds is not None:
            values["elapsed_seconds"] = max(
                0.0,
                float(elapsed_seconds),
            )

        if message is not None:
            values["message"] = str(message)

        if current_output is not None:
            values["current_output"] = str(current_output)

        self.job.update(**values)
