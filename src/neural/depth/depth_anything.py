"""Depth Anything V2 Small (INT8 Quantized) Runner (~27.3 MB ONNX).

Estimates continuous relative depth field D(x, y) in [0.0, 1.0] (1.0=near, 0.0=distant).
Enables depth-aware spatial background simplification (MTH-07) and cinematic lighting (MTH-09).
Supports graceful classical fallback using perspective gradient and focus proxy.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np

from ..runtime import ONNXRuntimeManager
from ..registry import ModelRegistry


class DepthAnythingRunner:
    """Runner for Depth Anything V2 Small with perspective-gradient classical fallback."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        device_preference: str = "auto",
    ):
        self.spec = ModelRegistry.get_spec("depth_anything_v2_small_int8")
        self.model_path = ModelRegistry.get_model_path("depth_anything_v2_small_int8", models_dir)
        self.session = None
        if self.model_path:
            self.session = ONNXRuntimeManager.create_session(self.model_path, device_preference)

    @property
    def is_neural_ready(self) -> bool:
        return self.session is not None

    def estimate_depth(
        self,
        rgb_uint8: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates relative scene depth field D(x, y) in [0.0, 1.0].
        Returns:
            (depth_map, latency_ms): Depth has shape (H, W, 1), float32.
        """
        h, w = rgb_uint8.shape[:2]
        start_t = time.perf_counter()

        if self.session is not None:
            try:
                inp_size = self.spec.input_resolution if self.spec else (518, 518)
                resized = cv2.resize(rgb_uint8, inp_size, interpolation=cv2.INTER_AREA)
                norm = resized.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (norm - mean) / std
                blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

                inp_name = self.session.get_inputs()[0].name
                out_name = self.session.get_outputs()[0].name
                raw_out = self.session.run([out_name], {inp_name: blob})[0]

                # raw_out shape: (1, 518, 518) or (1, 1, 518, 518)
                depth_raw = raw_out.squeeze()
                d_min = float(np.min(depth_raw))
                d_max = float(np.max(depth_raw))
                if (d_max - d_min) > 1e-4:
                    depth_norm = (depth_raw - d_min) / (d_max - d_min)
                else:
                    depth_norm = np.zeros_like(depth_raw)

                depth = cv2.resize(depth_norm, (w, h), interpolation=cv2.INTER_LINEAR)
                depth = np.clip(depth, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)
                latency = (time.perf_counter() - start_t) * 1000.0
                return depth, latency
            except Exception:
                pass  # Fall through to classical fallback

        # Graceful Classical Fallback: Perspective vertical gradient + edge focus proxy
        depth = self._classical_depth_fallback(rgb_uint8)
        latency = (time.perf_counter() - start_t) * 1000.0
        return depth, latency

    @staticmethod
    def _classical_depth_fallback(rgb: np.ndarray) -> np.ndarray:
        """Perspective linear vertical ramp with high-frequency detail focus proxy."""
        h, w = rgb.shape[:2]
        # Ground plane perspective prior: bottom of frame is nearer than top
        y_ramp = np.linspace(0.15, 0.95, h, dtype=np.float32)[:, np.newaxis]
        perspective = np.tile(y_ramp, (1, w))

        # Local focus proxy: high-frequency variance indicates nearer subject
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lap_smooth = cv2.GaussianBlur(lap, (25, 25), 0)
        lap_norm = np.clip(lap_smooth / max(1.0, float(np.percentile(lap_smooth, 95))), 0.0, 1.0)

        depth = perspective * 0.65 + lap_norm * 0.35
        return np.clip(depth[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)
