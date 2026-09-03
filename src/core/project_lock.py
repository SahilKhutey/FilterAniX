from __future__ import annotations

import os
from pathlib import Path

import psutil


class ProjectLockError(RuntimeError):
    pass


class ProjectLock:

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root)
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.lock_path = self.root / ".render.lock"

    def acquire(self):
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text(
                    encoding="utf-8"
                )
                pid = int(content.strip())
                if self._process_exists(pid) and pid != os.getpid():
                    raise ProjectLockError(
                        f"Project is already being rendered "
                        f"by process {pid}."
                    )
            except ValueError:
                pass

            self.lock_path.unlink(
                missing_ok=True
            )

        self.lock_path.write_text(
            str(os.getpid()),
            encoding="utf-8",
        )
        return self

    def release(self):
        self.lock_path.unlink(
            missing_ok=True
        )

    def __enter__(self):
        return self.acquire()

    def __exit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        self.release()

    @staticmethod
    def _process_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            return psutil.pid_exists(pid)
        except Exception:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
