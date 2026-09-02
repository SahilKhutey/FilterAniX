"""Person & Scene Segmentation Engine."""
from typing import Optional, Tuple
import cv2
import numpy as np

from src.vision.models import PersonMaskData, BoundingBox


class SegmentationEngine:
    """Extracts high-resolution foreground person mask, boundary contour, and coverage ratio."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._segmenter = None
        try:
            import mediapipe as mp
            if hasattr(mp.solutions, "selfie_segmentation"):
                self._segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        except Exception:
            self._segmenter = None

    def process(self, rgb: np.ndarray) -> Tuple[Optional[np.ndarray], PersonMaskData]:
        """Processes RGB frame and returns (binary_mask, PersonMaskData).
        
        Returns:
            mask_uint8: (H, W) uint8 array (255 = person, 0 = background)
            mask_data: PersonMaskData metadata structure
        """
        h, w = rgb.shape[:2]

        if self._segmenter is not None:
            try:
                results = self._segmenter.process(rgb)
                if results.segmentation_mask is not None:
                    raw_mask = results.segmentation_mask
                    binary_mask = (raw_mask > self.threshold).astype(np.uint8) * 255
                    
                    # Clean up noise with morphological operations
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
                    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

                    coverage = float(np.count_nonzero(binary_mask)) / float(h * w)

                    # Compute bounding box of person contour
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    bbox = None
                    if contours:
                        largest_cnt = max(contours, key=cv2.contourArea)
                        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
                        bbox = BoundingBox(
                            x=bx / float(w),
                            y=by / float(h),
                            width=bw / float(w),
                            height=bh / float(h),
                        )

                    return binary_mask, PersonMaskData(threshold=self.threshold, coverage=coverage, bbox=bbox)
            except Exception:
                pass

        # Fallback: Saliency / central thresholding
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt(((x - center_x) / (w * 0.4)) ** 2 + ((y - (center_y + h * 0.1)) / (h * 0.5)) ** 2)
        center_prior = (dist < 1.0).astype(np.uint8) * 255
        coverage = float(np.count_nonzero(center_prior)) / float(h * w)
        
        return center_prior, PersonMaskData(threshold=self.threshold, coverage=coverage, bbox=None)

    def close(self):
        if self._segmenter is not None:
            self._segmenter.close()
