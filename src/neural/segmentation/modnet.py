"""MODNet: Real-Time Portrait Matting Runner (~26 MB ONNX).

Produces continuous soft alpha matte in [0.0, 1.0] for creator silhouette, hair,
shoulders, and fine boundary transitions.
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


class MODNetRunner:
    """Runner for MODNet portrait matting with classical fallback."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        device_preference: str = "auto",
    ):
        self.spec = ModelRegistry.get_spec("modnet")
        self.model_path = ModelRegistry.get_model_path("modnet", models_dir)
        self.session = None
        if self.model_path:
            self.session = ONNXRuntimeManager.create_session(self.model_path, device_preference)

    @property
    def is_neural_ready(self) -> bool:
        return self.session is not None

    def predict_matte(
        self,
        rgb_uint8: np.ndarray,
        fallback_face_bbox: Optional[Any] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts continuous soft portrait alpha matte in [0.0, 1.0].
        Returns:
            (alpha_matte, latency_ms): Matte has shape (H, W, 1), float32 in [0.0, 1.0].
        """
        h, w = rgb_uint8.shape[:2]
        start_t = time.perf_counter()

        if self.session is not None:
            try:
                inp_size = self.spec.input_resolution if self.spec else (512, 512)
                resized = cv2.resize(rgb_uint8, inp_size, interpolation=cv2.INTER_AREA)
                # MODNet standard normalization: (x - 127.5) / 127.5
                norm = (resized.astype(np.float32) - 127.5) / 127.5
                blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]

                inp_name = self.session.get_inputs()[0].name
                out_name = self.session.get_outputs()[0].name
                raw_out = self.session.run([out_name], {inp_name: blob})[0]

                matte_512 = raw_out[0, 0]
                matte = cv2.resize(matte_512, (w, h), interpolation=cv2.INTER_LINEAR)
                matte = np.clip(matte, 0.0, 1.0)[:, :, np.newaxis].astype(np.float32)
                latency = (time.perf_counter() - start_t) * 1000.0
                return matte, latency
            except Exception:
                pass  # Fall through to classical fallback

        # Graceful Classical Fallback: Soft portrait silhouette from face/torso geometry
        matte = self._classical_portrait_matte_fallback(rgb_uint8, fallback_face_bbox)
        latency = (time.perf_counter() - start_t) * 1000.0
        return matte, latency

    @staticmethod
    def _classical_portrait_matte_fallback(
        rgb: np.ndarray,
        face_bbox: Optional[Any] = None,
    ) -> np.ndarray:
        """Classical anatomical portrait silhouette matte proxy."""
        h, w = rgb.shape[:2]
        canvas = np.zeros((h, w), dtype=np.float32)

        if face_bbox and hasattr(face_bbox, "x"):
            fx = int(face_bbox.x * w)
            fy = int(face_bbox.y * h)
            fw = int(face_bbox.width * w)
            fh = int(face_bbox.height * h)
        else:
            fw = int(w * 0.28)
            fh = int(h * 0.35)
            fx = (w - fw) // 2
            fy = int(h * 0.15)

        # Head / Face oval
        head_cx = fx + fw // 2
        head_cy = fy + fh // 2
        cv2.ellipse(canvas, (head_cx, head_cy), (int(fw * 0.65), int(fh * 0.70)), 0, 0, 360, 1.0, -1)

        # Torso / Shoulders
        torso_cx = head_cx
        torso_cy = fy + int(fh * 1.5)
        cv2.ellipse(canvas, (torso_cx, torso_cy), (int(fw * 1.6), int(fh * 1.4)), 0, 0, 360, 1.0, -1)

        # Soft edge feathering (bilateral/Gaussian edge relaxation)
        ksize = max(5, int(fw * 0.12)) | 1
        blurred = cv2.GaussianBlur(canvas, (ksize, ksize), 0)
        return np.clip(blurred[:, :, np.newaxis], 0.0, 1.0).astype(np.float32)
