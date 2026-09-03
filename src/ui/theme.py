from __future__ import annotations

import gradio as gr

# Custom Studio CSS
STUDIO_CUSTOM_CSS = """
:root {
    --bg-primary: #0D1117;
    --bg-secondary: #161B22;
    --bg-tertiary: #21262D;
    --border-color: #30363D;
    --text-primary: #E6EDF3;
    --text-muted: #8B949E;
    --accent-blue: #58A6FF;
    --accent-green: #3FB950;
    --accent-amber: #D29922;
    --accent-red: #F85149;
    --accent-purple: #BC8CFF;
}

body, .gradio-container {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

/* Studio Header */
.studio-header {
    background: linear-gradient(90deg, #161B22 0%, #1c2330 100%);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.studio-title {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--text-primary);
}

.studio-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 600;
}

.badge-ready { background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border: 1px solid rgba(63, 185, 80, 0.3); }
.badge-running { background: rgba(210, 153, 34, 0.15); color: var(--accent-amber); border: 1px solid rgba(210, 153, 34, 0.3); }
.badge-error { background: rgba(248, 81, 73, 0.15); color: var(--accent-red); border: 1px solid rgba(248, 81, 73, 0.3); }
.badge-info { background: rgba(88, 166, 255, 0.15); color: var(--accent-blue); border: 1px solid rgba(88, 166, 255, 0.3); }

/* Status Cards */
.status-card {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px 18px;
    min-height: 100px;
}

.status-card-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-muted);
    margin-bottom: 6px;
    font-weight: 600;
}

.status-card-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}

.status-card-sub {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* Pipeline Stepper */
.pipeline-stepper {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 10px;
    margin-bottom: 10px;
    font-family: monospace;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Status Bar / Telemetry Strip */
.studio-status-bar {
    background: #161B22;
    border-top: 1px solid var(--border-color);
    padding: 10px 18px;
    border-radius: 6px;
    margin-top: 15px;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
    font-size: 0.88rem;
    color: var(--text-primary);
    display: flex;
    justify-content: space-between;
}

/* Error Card */
.error-panel {
    background-color: rgba(248, 81, 73, 0.08);
    border: 1px solid var(--accent-red);
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
}

/* Notification item */
.notif-item {
    padding: 6px 10px;
    border-bottom: 1px solid rgba(48, 54, 61, 0.5);
    font-size: 0.86rem;
}
"""


def get_studio_theme() -> gr.themes.Base:
    """Returns dark production studio theme for FilterAniX."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        text_size=gr.themes.sizes.text_sm,
    )
