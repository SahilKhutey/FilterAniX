from __future__ import annotations

import gradio as gr
from src.ui.state import ui_state_manager


def format_notification_text() -> str:
    """Formats recent studio notifications as bulleted log text."""
    notifs = ui_state_manager.get_notifications(12)
    if not notifs:
        return "• Studio ready. No events recorded."
    lines = []
    for n in notifs:
        icon = "•"
        if n.level == "success":
            icon = "✓"
        elif n.level == "error":
            icon = "❌"
        elif n.level == "warning":
            icon = "⚠"
        lines.append(f"[{n.formatted_time()}] {icon} {n.message}")
    return "\n".join(lines)


def create_notifications_feed() -> gr.Textbox:
    """Renders the notification center feed."""
    feed = gr.Textbox(
        label="Notification Center & Event Feed",
        value=format_notification_text,
        lines=6,
        interactive=False,
    )
    return feed
