"""Frame data structures and representations for FilterAniX."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class FrameData:
    """Rich multi-layer frame representation passing through the FilterAniX pipeline."""
    frame_index: int
    timestamp_sec: float
    rgb: np.ndarray  # Shape (H, W, 3), uint8 (0-255)
    
    # Vision & Geometry Layers
    person_mask: Optional[np.ndarray] = None      # Shape (H, W), float32 [0.0, 1.0]
    background_mask: Optional[np.ndarray] = None  # Shape (H, W), float32 [0.0, 1.0]
    face_mask: Optional[np.ndarray] = None        # Shape (H, W), float32 [0.0, 1.0]
    depth_map: Optional[np.ndarray] = None        # Shape (H, W), float32 [0.0, 1.0]
    
    # Motion & Optical Flow
    forward_flow: Optional[np.ndarray] = None     # Shape (H, W, 2), float32 (dx, dy)
    backward_flow: Optional[np.ndarray] = None    # Shape (H, W, 2), float32 (dx, dy)
    occlusion_mask: Optional[np.ndarray] = None   # Shape (H, W), float32 [0.0, 1.0] (1.0 = visible, 0.0 = occluded)
    
    # Landmarks
    face_landmarks: Optional[np.ndarray] = None   # Shape (N, 2 or 3), normalized or pixel coordinates
    pose_landmarks: Optional[np.ndarray] = None   # Shape (N, 2 or 3)
    left_hand_landmarks: Optional[np.ndarray] = None
    right_hand_landmarks: Optional[np.ndarray] = None
    
    # Rendered & Stylized Layers
    stylized_fg: Optional[np.ndarray] = None      # Shape (H, W, 3), uint8
    stylized_bg: Optional[np.ndarray] = None      # Shape (H, W, 3), uint8
    line_art: Optional[np.ndarray] = None         # Shape (H, W, 3) or (H, W), uint8
    warped_prior_fg: Optional[np.ndarray] = None  # Shape (H, W, 3), uint8 (flow-warped from previous frame)
    
    # Final Post-Processed Frame
    final_composite: Optional[np.ndarray] = None  # Shape (H, W, 3), uint8
    
    # Diagnostics & Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def height(self) -> int:
        return self.rgb.shape[0]
        
    @property
    def width(self) -> int:
        return self.rgb.shape[1]
        
    def copy_shallow(self) -> "FrameData":
        return FrameData(
            frame_index=self.frame_index,
            timestamp_sec=self.timestamp_sec,
            rgb=self.rgb,
            person_mask=self.person_mask,
            background_mask=self.background_mask,
            face_mask=self.face_mask,
            depth_map=self.depth_map,
            forward_flow=self.forward_flow,
            backward_flow=self.backward_flow,
            occlusion_mask=self.occlusion_mask,
            face_landmarks=self.face_landmarks,
            pose_landmarks=self.pose_landmarks,
            left_hand_landmarks=self.left_hand_landmarks,
            right_hand_landmarks=self.right_hand_landmarks,
            stylized_fg=self.stylized_fg,
            stylized_bg=self.stylized_bg,
            line_art=self.line_art,
            warped_prior_fg=self.warped_prior_fg,
            final_composite=self.final_composite,
            metadata=self.metadata.copy(),
        )
