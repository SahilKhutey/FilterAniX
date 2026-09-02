"""Optical flow estimation and backward/forward warping."""
from typing import Optional, Tuple
import cv2
import numpy as np


class OpticalFlowEstimator:
    """Computes dense optical flow fields and performs forward/backward warping."""

    def __init__(self, algorithm: str = "dis", preset: str = "medium"):
        self.algorithm = algorithm.lower()
        self.preset = preset.lower()
        self._dis = None

        if self.algorithm == "dis":
            self._dis = cv2.DISOpticalFlow_create(
                cv2.DISOPTICAL_FLOW_PRESET_MEDIUM if preset == "medium"
                else cv2.DISOPTICAL_FLOW_PRESET_FAST if preset == "fast"
                else cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
            )

    def compute_flow(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """Computes dense optical flow (curr -> prev or prev -> curr depending on input order).
        
        Returns:
            flow: Shape (H, W, 2), float32 with (dx, dy)
        """
        if self._dis is not None:
            flow = self._dis.calc(prev_gray, curr_gray, None)
        else:
            # Fallback to Farneback
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
        return flow

    def compute_bidirectional_flow(
        self, prev_rgb: np.ndarray, curr_rgb: np.ndarray, occlusion_threshold: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes forward flow (prev -> curr), backward flow (curr -> prev), and occlusion mask.
        
        Returns:
            forward_flow: (H, W, 2)
            backward_flow: (H, W, 2)
            occlusion_mask: (H, W) float32 [0.0 = occluded, 1.0 = visible/consistent]
        """
        prev_gray = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)

        forward_flow = self.compute_flow(prev_gray, curr_gray)
        backward_flow = self.compute_flow(curr_gray, prev_gray)

        # Forward-backward consistency check to find occlusions
        # Warp backward flow with forward flow
        h, w = prev_gray.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Where pixels in 'curr' came from in 'prev'
        map_x = (grid_x + backward_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + backward_flow[..., 1]).astype(np.float32)
        
        # Warp forward flow onto curr coords
        forward_warped = cv2.remap(
            forward_flow, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        
        # Consistency error: backward_flow + forward_warped should be near zero vector
        diff = backward_flow + forward_warped
        sq_error = np.sum(diff ** 2, axis=-1)
        magnitude = np.sum(backward_flow ** 2, axis=-1) + np.sum(forward_warped ** 2, axis=-1)
        
        # Occlusion score
        threshold_sq = occlusion_threshold ** 2
        is_consistent = (sq_error < (0.01 * magnitude + threshold_sq)).astype(np.float32)
        
        # Smooth occlusion mask slightly
        occlusion_mask = cv2.GaussianBlur(is_consistent, (5, 5), 0)
        
        return forward_flow, backward_flow, occlusion_mask

    @staticmethod
    def warp_frame(image: np.ndarray, backward_flow: np.ndarray) -> np.ndarray:
        """Warps previous image 'I_{t-1}' forward to current time 't' using backward flow (curr -> prev).
        
        Args:
            image: Image at t-1 (H, W, C) or (H, W)
            backward_flow: Flow from curr (t) to prev (t-1), (H, W, 2)
            
        Returns:
            Warped image aligned with current frame geometry.
        """
        h, w = image.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + backward_flow[..., 0]).astype(np.float32)
        map_y = (grid_y + backward_flow[..., 1]).astype(np.float32)

        warped = cv2.remap(
            image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        return warped
