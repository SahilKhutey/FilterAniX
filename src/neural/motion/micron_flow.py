"""Micron-Flow: Lightweight Neural Optical Flow Runner (~2.10 MB ONNX, 522K parameters).

Estimates dense 2D motion vector field (dx, dy) between consecutive frames.
Feeds directly into MTH-10 Temporal Field for warp-stabilization.
Supports graceful classical fallback to OpenCV Farneback optical flow.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ..runtime import ONNXRuntimeManager
from ..registry import ModelRegistry


class MicronFlowRunner:
    """Runner for Micron-Flow neural optical flow with Farneback classical fallback."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        device_preference: str = "auto",
    ):
        self.spec = ModelRegistry.get_spec("micron_flow")
        self.model_path = ModelRegistry.get_model_path("micron_flow", models_dir)
        self.session = None
        if self.model_path:
            self.session = ONNXRuntimeManager.create_session(self.model_path, device_preference)

    @property
    def is_neural_ready(self) -> bool:
        return self.session is not None

    def estimate_flow(
        self,
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates dense optical flow field from prev_rgb to curr_rgb.
        Returns:
            (flow_field, latency_ms): Flow has shape (H, W, 2), float32.
        """
        h, w = curr_rgb.shape[:2]
        start_t = time.perf_counter()

        if self.session is not None:
            try:
                inp_size = self.spec.input_resolution if self.spec else (256, 256)
                p_small = cv2.resize(prev_rgb, inp_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
                c_small = cv2.resize(curr_rgb, inp_size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

                # Concatenate along channel axis: (1, 6, 256, 256)
                blob = np.concatenate([p_small, c_small], axis=-1)
                blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

                inp_name = self.session.get_inputs()[0].name
                out_name = self.session.get_outputs()[0].name
                raw_flow = self.session.run([out_name], {inp_name: blob})[0]

                # raw_flow shape: (1, 2, 256, 256)
                flow_small = np.transpose(raw_flow[0], (1, 2, 0))
                # Scale flow vectors to original frame dimensions
                scale_x = float(w) / float(inp_size[0])
                scale_y = float(h) / float(inp_size[1])
                flow = cv2.resize(flow_small, (w, h), interpolation=cv2.INTER_LINEAR)
                flow[:, :, 0] *= scale_x
                flow[:, :, 1] *= scale_y

                latency = (time.perf_counter() - start_t) * 1000.0
                return flow.astype(np.float32), latency
            except Exception:
                pass  # Fall through to classical fallback

        # Graceful Classical Fallback: OpenCV Farneback Optical Flow
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        latency = (time.perf_counter() - start_t) * 1000.0
        return flow.astype(np.float32), latency
