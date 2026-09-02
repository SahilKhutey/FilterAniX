"""Optical Flow-Guided Warp Blending for Temporal Consistency."""
from typing import Optional, Tuple
import cv2
import numpy as np
from filteranix.core.config import TemporalConfig
from filteranix.vision.optical_flow import OpticalFlowEstimator


class TemporalWarpBlender:
    """Eliminates frame-to-frame flickering by warping stylized prior frames along dense optical flow fields."""

    def __init__(self, config: Optional[TemporalConfig] = None):
        self.config = config or TemporalConfig()
        self.flow_estimator = OpticalFlowEstimator()

    def blend_with_prior(
        self,
        curr_stylized_rgb: np.ndarray,
        prev_stylized_rgb: np.ndarray,
        curr_raw_rgb: np.ndarray,
        prev_raw_rgb: np.ndarray,
        person_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Warps previous stylized frame into current frame geometry and blends with occlusion awareness.
        
        Returns:
            blended_rgb: Temporally coherent stylized frame (H, W, 3)
            warped_prior: The raw warped prior frame (H, W, 3)
        """
        # Compute bidirectional optical flow and occlusion map between raw frames
        forward_flow, backward_flow, occlusion_mask = self.flow_estimator.compute_bidirectional_flow(
            prev_raw_rgb, curr_raw_rgb, occlusion_threshold=self.config.occlusion_threshold
        )

        # Warp the previous stylized output along backward flow (curr -> prev)
        warped_prior = self.flow_estimator.warp_frame(prev_stylized_rgb, backward_flow)

        # Blend weight: high in consistent/visible areas, low in occluded/newly disoccluded areas
        effective_alpha = self.config.warp_blend_alpha * occlusion_mask
        if person_mask is not None:
            effective_alpha = effective_alpha * person_mask

        alpha_3d = effective_alpha[..., np.newaxis]
        
        curr_f = curr_stylized_rgb.astype(np.float32)
        prior_f = warped_prior.astype(np.float32)
        
        blended = (1.0 - alpha_3d) * curr_f + alpha_3d * prior_f
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)

        return blended_uint8, warped_prior
