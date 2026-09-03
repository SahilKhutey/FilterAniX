"""FilterAniX Neural Assistance Manager.

Orchestrates pluggable multi-rate neural inference (Enable / Disable feature)
and injects continuous semantic control fields into FrameVisionData:
  - Portrait Matting (MODNet / U²-Netp) -> MTH-07
  - Relative Depth (Depth Anything V2 INT8) -> MTH-07 / MTH-09
  - Optical Flow (Micron-Flow) -> MTH-10
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np

from .config import NeuralAssistConfig
from .memory import NeuralMemoryTracker
from .segmentation.modnet import MODNetRunner
from .segmentation.u2netp import U2NetpRunner
from .motion.micron_flow import MicronFlowRunner
from .depth.depth_anything import DepthAnythingRunner

logger = logging.getLogger("filteranix.neural.manager")


class NeuralAssistManager:
    """Manages multi-rate execution of assistive neural models with temporal field propagation."""

    def __init__(
        self,
        config: Optional[NeuralAssistConfig] = None,
    ):
        self.config = config or NeuralAssistConfig()
        self.memory_tracker = NeuralMemoryTracker(self.config.models_dir)

        # Lazy runner initialization
        self._modnet: Optional[MODNetRunner] = None
        self._u2netp: Optional[U2NetpRunner] = None
        self._micron_flow: Optional[MicronFlowRunner] = None
        self._depth: Optional[DepthAnythingRunner] = None

        # Temporal cache for multi-rate propagation
        self._frame_count: int = 0
        self._prev_rgb: Optional[np.ndarray] = None
        self._last_matte: Optional[np.ndarray] = None
        self._last_depth: Optional[np.ndarray] = None
        self._last_flow: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets frame counter and temporal field buffers across cuts or scene changes."""
        self._frame_count = 0
        self._prev_rgb = None
        self._last_matte = None
        self._last_depth = None
        self._last_flow = None

    def _get_modnet(self) -> MODNetRunner:
        if self._modnet is None:
            self._modnet = MODNetRunner(self.config.models_dir, self.config.device_preference)
        return self._modnet

    def _get_u2netp(self) -> U2NetpRunner:
        if self._u2netp is None:
            self._u2netp = U2NetpRunner(self.config.models_dir, self.config.device_preference)
        return self._u2netp

    def _get_micron_flow(self) -> MicronFlowRunner:
        if self._micron_flow is None:
            self._micron_flow = MicronFlowRunner(self.config.models_dir, self.config.device_preference)
        return self._micron_flow

    def _get_depth(self) -> DepthAnythingRunner:
        if self._depth is None:
            self._depth = DepthAnythingRunner(self.config.models_dir, self.config.device_preference)
        return self._depth

    def process_frame(
        self,
        rgb_uint8: np.ndarray,
        vision_data: Any,
        scene_cut: bool = False,
    ) -> Any:
        """
        Executes assistive neural models according to multi-rate schedule and
        injects neural fields into vision_data.
        If config.enabled is False, immediately returns vision_data untouched.
        """
        if not self.config.enabled:
            return vision_data

        if scene_cut:
            self.reset()

        h, w = rgb_uint8.shape[:2]
        idx = self._frame_count
        self._frame_count += 1

        face_bbox = None
        if hasattr(vision_data, "faces") and vision_data.faces:
            face_bbox = getattr(vision_data.faces[0], "bbox", None)

        telemetry: dict[str, Any] = {"neural_enabled": True}

        # 1. Optical Flow Estimation (Every frame if prev frame available)
        flow_field = None
        if self.config.use_flow and self._prev_rgb is not None:
            flow_runner = self._get_micron_flow()
            flow_field, f_lat = flow_runner.estimate_flow(self._prev_rgb, rgb_uint8)
            self.memory_tracker.record_latency("micron_flow", f_lat)
            telemetry["micron_flow_latency_ms"] = f_lat
            self._last_flow = flow_field
        elif self._last_flow is not None:
            flow_field = self._last_flow

        # 2. Portrait Matting / Salient Segmentation (Every N frames)
        matte_field = None
        run_matting = (idx % max(1, self.config.matting_interval) == 0) or (self._last_matte is None)

        if self.config.use_matting:
            if run_matting:
                modnet = self._get_modnet()
                matte_field, m_lat = modnet.predict_matte(rgb_uint8, face_bbox)
                self.memory_tracker.record_latency("modnet", m_lat)
                telemetry["modnet_latency_ms"] = m_lat
                self._last_matte = matte_field
            else:
                # Multi-rate propagation: warp previous matte forward with optical flow
                if self._last_matte is not None and flow_field is not None:
                    matte_field = self._warp_field(self._last_matte, flow_field)
                else:
                    matte_field = self._last_matte
        elif self.config.use_segmentation:
            if run_matting:
                u2netp = self._get_u2netp()
                matte_field, u_lat = u2netp.predict_mask(rgb_uint8, face_bbox)
                self.memory_tracker.record_latency("u2netp", u_lat)
                telemetry["u2netp_latency_ms"] = u_lat
                self._last_matte = matte_field
            else:
                matte_field = self._last_matte

        # 3. Depth Estimation (Every M frames)
        depth_field = None
        run_depth = (idx % max(1, self.config.depth_interval) == 0) or (self._last_depth is None)

        if self.config.use_depth:
            if run_depth:
                depth_runner = self._get_depth()
                depth_field, d_lat = depth_runner.estimate_depth(rgb_uint8)
                self.memory_tracker.record_latency("depth_anything_v2_small_int8", d_lat)
                telemetry["depth_latency_ms"] = d_lat
                self._last_depth = depth_field
            else:
                depth_field = self._last_depth

        # Store current frame for next flow step
        self._prev_rgb = rgb_uint8.copy()

        # Inject continuous fields into vision_data
        setattr(vision_data, "neural_matte", matte_field)
        setattr(vision_data, "neural_depth", depth_field)
        setattr(vision_data, "neural_flow", flow_field)
        setattr(vision_data, "neural_telemetry", telemetry)

        return vision_data

    @staticmethod
    def _warp_field(field_map: np.ndarray, flow_uv: np.ndarray) -> np.ndarray:
        """Warps a 2D scalar field forward using optical flow vectors."""
        h, w = field_map.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x - flow_uv[:, :, 0]).astype(np.float32)
        map_y = (grid_y - flow_uv[:, :, 1]).astype(np.float32)
        warped = cv2.remap(
            field_map,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        if warped.ndim == 2:
            warped = warped[:, :, np.newaxis]
        return np.clip(warped, 0.0, 1.0)
