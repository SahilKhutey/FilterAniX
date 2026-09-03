from __future__ import annotations

import gradio as gr
from src.core.hardware import get_hardware_report


def create_header():
    """Renders the top production studio header with real-time status and hardware badges."""
    hw = get_hardware_report()
    gpu_label = hw.gpu_name if hw.cuda_available else "CPU Execution"
    cpu_cores = hw.cpu_cores

    with gr.Row(elem_classes=["studio-header"]):
        with gr.Column(scale=2):
            gr.Markdown(
                """
                # 🎬 FILTERANIX STUDIO <span style="font-size:0.9rem; font-weight:normal; color:#8B949E;">v2.0 Production Control Center</span>
                """
            )
        with gr.Column(scale=1):
            with gr.Row():
                gr.HTML(
                    f"""
                    <div style="display:flex; gap:10px; justify-content:flex-end; align-items:center;">
                        <span class="studio-badge badge-ready">● SYSTEM READY</span>
                        <span class="studio-badge badge-info">⚡ {gpu_label} ({cpu_cores} Cores)</span>
                    </div>
                    """
                )
