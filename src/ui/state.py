from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class UIState(str, Enum):
    IDLE = "IDLE"
    INPUT_READY = "INPUT_READY"
    ANALYZING = "ANALYZING"
    READY_TO_RENDER = "READY_TO_RENDER"
    RENDERING = "RENDERING"
    PAUSED = "PAUSED"
    COMPOSING = "COMPOSING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


@dataclass
class UINotification:
    """Timestamped studio notification message."""
    message: str
    level: str = "info"  # "info", "success", "warning", "error"
    timestamp: float = field(default_factory=time.time)

    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class UIStateManager:
    """
    Manages explicit UI states, button interactivity rules, and studio notifications.
    Ensures UI buttons and controls remain completely consistent across states.
    """

    def __init__(self):
        self._state: UIState = UIState.IDLE
        self._active_job_id: str = ""
        self._active_project_name: str = ""
        self._notifications: List[UINotification] = []
        self._last_error: Optional[str] = None
        self._error_stage: Optional[str] = None
        self._error_frame: Optional[int] = None
        self._error_details: Dict[str, Any] = {}

    @property
    def current_state(self) -> UIState:
        return self._state

    def set_state(self, new_state: UIState) -> None:
        self._state = new_state

    def set_active_job(self, job_id: str):
        self._active_job_id = job_id

    @property
    def active_job_id(self) -> str:
        return self._active_job_id

    def set_active_project(self, name: str):
        self._active_project_name = name

    @property
    def active_project_name(self) -> str:
        return self._active_project_name

    def record_error(
        self,
        error_message: str,
        stage: str = "",
        frame: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self._state = UIState.ERROR
        self._last_error = error_message
        self._error_stage = stage
        self._error_frame = frame
        self._error_details = details or {}
        self.add_notification(f"Rendering error in {stage or 'pipeline'}: {error_message}", level="error")

    def clear_error(self):
        self._last_error = None
        self._error_stage = None
        self._error_frame = None
        self._error_details.clear()
        if self._state == UIState.ERROR:
            self._state = UIState.IDLE

    def get_error_info(self) -> Dict[str, Any]:
        return {
            "error": self._last_error,
            "stage": self._error_stage,
            "frame": self._error_frame,
            "details": self._error_details,
        }

    def add_notification(self, message: str, level: str = "info") -> None:
        notif = UINotification(message=message, level=level)
        self._notifications.append(notif)
        # Keep recent 100 notifications
        if len(self._notifications) > 100:
            self._notifications.pop(0)

    def get_notifications(self, limit: int = 15) -> List[UINotification]:
        return self._notifications[-limit:]

    def get_button_states(self, state: Optional[UIState] = None) -> Dict[str, bool]:
        """
        Computes interactive enabled/disabled states for pipeline buttons:
        - start_enabled
        - pause_enabled
        - stop_enabled
        - resume_enabled
        - retry_enabled
        """
        st = state or self._state

        if st == UIState.RENDERING or st == UIState.ANALYZING or st == UIState.COMPOSING or st == UIState.VALIDATING:
            return {
                "start_enabled": False,
                "pause_enabled": True,
                "stop_enabled": True,
                "resume_enabled": False,
                "retry_enabled": False,
            }
        elif st == UIState.PAUSED:
            return {
                "start_enabled": False,
                "pause_enabled": False,
                "stop_enabled": True,
                "resume_enabled": True,
                "retry_enabled": False,
            }
        elif st == UIState.ERROR:
            return {
                "start_enabled": True,
                "pause_enabled": False,
                "stop_enabled": True,
                "resume_enabled": True,
                "retry_enabled": True,
            }
        else:
            # IDLE, INPUT_READY, READY_TO_RENDER, COMPLETE
            return {
                "start_enabled": True,
                "pause_enabled": False,
                "stop_enabled": False,
                "resume_enabled": False,
                "retry_enabled": False,
            }


# Singleton manager instance
ui_state_manager = UIStateManager()
