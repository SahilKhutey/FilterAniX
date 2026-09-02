"""Person & Background Segmentation and Background Plate Generation."""
from typing import List, Optional, Tuple
import cv2
import numpy as np


class VideoSegmenter:
    """Segments creator foreground (person, hands, dynamic clothing) from static background."""

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe
        self._mp_segmenter = None
        
        if self.use_mediapipe:
            try:
                import mediapipe as mp
                # Try mediapipe selfie segmentation
                if hasattr(mp.solutions, "selfie_segmentation"):
                    self._mp_segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
            except Exception:
                self._mp_segmenter = None

    def segment_frame(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Segments RGB frame into foreground (person) mask and background mask.
        
        Args:
            rgb: uint8 array (H, W, 3)
            
        Returns:
            person_mask: (H, W) float32 in [0.0, 1.0]
            bg_mask: (H, W) float32 in [0.0, 1.0]
        """
        h, w = rgb.shape[:2]
        
        if self._mp_segmenter is not None:
            try:
                results = self._mp_segmenter.process(rgb)
                if results.segmentation_mask is not None:
                    raw_mask = results.segmentation_mask.astype(np.float32)
                    # Clean up mask with soft thresholding and guided edge smoothing
                    raw_mask = cv2.GaussianBlur(raw_mask, (7, 7), 0)
                    person_mask = np.clip((raw_mask - 0.2) / 0.6, 0.0, 1.0)
                    bg_mask = 1.0 - person_mask
                    return person_mask, bg_mask
            except Exception:
                pass

        # Fallback: Robust saliency and central-prior GrabCut / threshold segmentation
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # Create center-focused elliptical prior (creator sitting at center desk)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt(((x - center_x) / (w * 0.4)) ** 2 + ((y - (center_y + h * 0.1)) / (h * 0.5)) ** 2)
        center_prior = np.clip(1.0 - dist_from_center, 0.0, 1.0).astype(np.float32)
        
        # Combine with edge & gradient energy
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_mag = grad_mag / (np.max(grad_mag) + 1e-5)
        
        raw_mask = 0.7 * center_prior + 0.3 * grad_mag
        raw_mask = cv2.GaussianBlur(raw_mask, (15, 15), 0)
        person_mask = np.clip((raw_mask - 0.3) / 0.4, 0.0, 1.0).astype(np.float32)
        bg_mask = 1.0 - person_mask
        
        return person_mask, bg_mask


class BackgroundPlateBuilder:
    """Builds a pristine, noise-free static background plate by accumulating background regions over time."""

    def __init__(self, target_frames: int = 30):
        self.target_frames = target_frames
        self.bg_accumulator: Optional[np.ndarray] = None
        self.weight_accumulator: Optional[np.ndarray] = None
        self.completed_plate: Optional[np.ndarray] = None

    def add_frame(self, rgb: np.ndarray, bg_mask: np.ndarray):
        """Accumulates clean background pixels weighted by the background probability mask."""
        if self.completed_plate is not None:
            return

        h, w = rgb.shape[:2]
        if self.bg_accumulator is None:
            self.bg_accumulator = np.zeros((h, w, 3), dtype=np.float32)
            self.weight_accumulator = np.zeros((h, w, 1), dtype=np.float32)

        weight = np.clip(bg_mask[..., np.newaxis], 0.0, 1.0)
        self.bg_accumulator += rgb.astype(np.float32) * weight
        self.weight_accumulator += weight

    def build(self, fallback_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Finalizes the static background plate, inpainting any areas that were constantly occluded."""
        if self.completed_plate is not None:
            return self.completed_plate

        if self.bg_accumulator is None and fallback_frame is not None:
            self.completed_plate = fallback_frame.copy()
            return self.completed_plate

        valid_mask = (self.weight_accumulator > 1e-3).astype(np.float32)
        safe_weights = np.maximum(self.weight_accumulator, 1e-3)
        plate = (self.bg_accumulator / safe_weights).astype(np.float32)

        # Inpaint occluded holes if necessary
        inpaint_mask = (valid_mask[..., 0] == 0).astype(np.uint8)
        if np.any(inpaint_mask) and fallback_frame is not None:
            plate_uint8 = np.clip(plate, 0, 255).astype(np.uint8)
            plate_uint8 = np.where(inpaint_mask[..., np.newaxis] > 0, fallback_frame, plate_uint8)
            plate = plate_uint8.astype(np.float32)
        elif np.any(inpaint_mask):
            plate_uint8 = np.clip(plate, 0, 255).astype(np.uint8)
            plate = cv2.inpaint(plate_uint8, inpaint_mask, 5, cv2.INPAINT_TELEA).astype(np.float32)

        self.completed_plate = np.clip(plate, 0, 255).astype(np.uint8)
        return self.completed_plate
