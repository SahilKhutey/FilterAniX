from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageState:
    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    output: Optional[str] = None

    def to_dict(self):
        data = asdict(self)
        data["status"] = self.status.value
        return data
