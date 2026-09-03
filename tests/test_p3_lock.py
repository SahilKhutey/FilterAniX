import os
import pytest
from src.core.project_lock import (
    ProjectLock,
    ProjectLockError,
)


def test_project_lock(tmp_path):
    with ProjectLock(tmp_path):
        assert (tmp_path / ".render.lock").exists()


def test_project_lock_cleans_dead_pid(tmp_path):
    lock_file = tmp_path / ".render.lock"
    lock_file.write_text("999999999", encoding="utf-8")  # Non-existent PID

    with ProjectLock(tmp_path):
        assert lock_file.exists()
        assert int(lock_file.read_text().strip()) == os.getpid()


def test_project_lock_rejects_active_foreign_process(tmp_path):
    import psutil
    # Find an active PID that is not us
    other_pid = None
    for p in psutil.process_iter(['pid']):
        if p.info['pid'] != os.getpid() and p.info['pid'] > 0:
            other_pid = p.info['pid']
            break

    if other_pid is not None:
        lock_file = tmp_path / ".render.lock"
        lock_file.write_text(str(other_pid), encoding="utf-8")

        with pytest.raises(ProjectLockError):
            with ProjectLock(tmp_path):
                pass
