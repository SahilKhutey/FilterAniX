"""Tests for UI State Machine and button interactivity rules."""
import pytest
from src.ui.state import UIState, UIStateManager, UINotification


def test_idle_state():
    mgr = UIStateManager()
    assert mgr.current_state == UIState.IDLE
    btn_states = mgr.get_button_states()
    assert btn_states["start_enabled"] is True
    assert btn_states["pause_enabled"] is False
    assert btn_states["stop_enabled"] is False
    assert btn_states["resume_enabled"] is False
    assert btn_states["retry_enabled"] is False


def test_rendering_state():
    mgr = UIStateManager()
    mgr.set_state(UIState.RENDERING)
    assert mgr.current_state == UIState.RENDERING
    btn_states = mgr.get_button_states()
    assert btn_states["start_enabled"] is False
    assert btn_states["pause_enabled"] is True
    assert btn_states["stop_enabled"] is True
    assert btn_states["resume_enabled"] is False
    assert btn_states["retry_enabled"] is False


def test_paused_state():
    mgr = UIStateManager()
    mgr.set_state(UIState.PAUSED)
    assert mgr.current_state == UIState.PAUSED
    btn_states = mgr.get_button_states()
    assert btn_states["start_enabled"] is False
    assert btn_states["pause_enabled"] is False
    assert btn_states["stop_enabled"] is True
    assert btn_states["resume_enabled"] is True
    assert btn_states["retry_enabled"] is False


def test_complete_state():
    mgr = UIStateManager()
    mgr.set_state(UIState.COMPLETE)
    assert mgr.current_state == UIState.COMPLETE
    btn_states = mgr.get_button_states()
    assert btn_states["start_enabled"] is True
    assert btn_states["pause_enabled"] is False
    assert btn_states["stop_enabled"] is False
    assert btn_states["resume_enabled"] is False


def test_error_state():
    mgr = UIStateManager()
    mgr.record_error("Failed in MTH-09", stage="MTH-09 Lighting Field", frame=151)
    assert mgr.current_state == UIState.ERROR
    err_info = mgr.get_error_info()
    assert err_info["error"] == "Failed in MTH-09"
    assert err_info["stage"] == "MTH-09 Lighting Field"
    assert err_info["frame"] == 151

    btn_states = mgr.get_button_states()
    assert btn_states["start_enabled"] is True
    assert btn_states["retry_enabled"] is True
    assert btn_states["stop_enabled"] is True

    # Clear error
    mgr.clear_error()
    assert mgr.current_state == UIState.IDLE
    assert mgr.get_error_info()["error"] is None


def test_notifications():
    mgr = UIStateManager()
    mgr.add_notification("Video loaded", level="success")
    mgr.add_notification("Rendering started", level="info")
    notifs = mgr.get_notifications()
    assert len(notifs) >= 2
    assert "Rendering started" in notifs[-1].message
