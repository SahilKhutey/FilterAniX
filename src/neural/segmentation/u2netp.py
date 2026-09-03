"""U²-Netp: Lightweight Salient Object Segmentation Runner (~4.36 MB ONNX).

Produces continuous salient subject probability field P(subject | pixel) in [0.0, 1.0].
Supports graceful classical fallback when model weights or ONNX runtime are absent.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple
import cv2
import numpy as np

from ..runtime import ONNXRuntimeManager
from ..registry import ModelRegistry


class U2NetpRunner:
    """Runner for U²-Netp salient object segmentation with classical fallback."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        device_preference: str = "auto",
    ):
        self.spec = ModelRegistry.get_spec("u2netp")
        self.model_path = ModelRegistry.get_model_path("u2netp", models_dir)
        self.session = None
        if self.model_path:
            self.session = ONNXRuntimeManager.create_session(self.model_path, device_preference)

    @property
    def is_neural_ready(self) -> bool:
        return self.session is not None

    def predict_mask(
        self,
        rgb_uint8: np.ndarray,
        fallback_hint_bbox: Optional[Any] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts salient foreground probability field P(subject | pixel).
        Returns:
            (mask_float32, latency_ms): Mask has shape (H, W, 1) with values in [0.0, 1.0].
        """
        h, w = rgb_uint8.shape[:2]
        start_t = time.perf_counter()

        if self.session is not None:
            try:
                # Preprocess: 320x320 NCHW normalized
                inp_size = self.spec.input_resolution if self.spec else (320, 320)
                resized = cv2.resize(rgb_uint8, inp_size, interpolation=cv2.INTER_AREA)
                norm = resized.astype(np.float32) / 255.0
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                norm = (norm - mean) / std
                blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

                inp_name = self.session.get_inputs()[0].name
                out_name = self.session.get_outputs()[0].name
                raw_out = self.session.run([out_name], {inp_name: blob})[0]

                # Postprocess: extract first feature map and apply min-max scaling
                pred = raw_out[0, 0]
                pred_min = float(np.min(pred))
                pred_max = float(np.max(pred))
                if (pred_max - pred_min) > 1e-4:
                    pred = (pred - pred_min) / (pred_max - pred_min)
                else:
                    pred = np.zeros_like(pred)

                mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.clip(mask, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)
                latency = (time.perf_counter() - start_t) * 1000.0
                return mask, latency
            except Exception:
                pass  # Fall through to classical fallback

        # Graceful Classical Fallback: Saliency proxy using center prior + gradient contrast
        mask = self._classical_saliency_fallback(rgb_uint8, fallback_hint_bbox)
        latency = (time.perf_counter() - start_t) * 1000.0
        return mask, latency

    @staticmethod
    def _classical_saliency_fallback(
        rgb: np.ndarray,
        hint_bbox: Optional[Any] = None,
    ) -> np.ndarray:
        """Classical center-prior and luminance variance foreground estimation."""
        h, w = rgb.shape[:2]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        # Center-weighted spatial prior
        y_coords, x_coords = np.ogrid[:h, :w]
        if hint_bbox and hasattr(hint_bbox, "x"):
            xc = int((hint_bbox.x + hint_bbox.width / 2.0) * w)
            yc = int((hint_bbox.y + hint_bbox.height / 2.0) * h)
            rx = max(w * 0.30, hint_bbox.width * w * 0.9)
            ry = max(h * 0.35, hint_bbox.height * h * 1.1)
        else:
            xc = w // 2
            yc = int(h * 0.52)
            rx = w * 0.38
            ry = h * 0.45

        dist_sq = ((x_coords - xc) / max(rx, 1.0)) ** 2 + ((y_coords - yc) / max(ry, 1.0)) ** 2
        prior = np.exp(-dist_sq * 1.4).astype(np.float32)

        # Contrast map
        diff = np.abs(gray.astype(np.float32) - float(np.mean(blurred))) / 255.0
        saliency = prior * 0.75 + diff * 0.25
        return np.clip(saliency[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)
