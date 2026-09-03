from __future__ import annotations

import cv2
import numpy as np


class MouthRenderer:
    """Renders mouth opening and anime cavity stylization matching viseme state."""

    STATES = {
        "closed": 0.0,
        "slightly_open": 0.25,
        "open": 0.55,
        "wide_open": 0.85,
    }

    def render(
        self,
        frame: np.ndarray,
        mouth_bbox: tuple[int, int, int, int] | None,
        viseme: str,
    ) -> np.ndarray:

        if mouth_bbox is None:
            return frame

        x1, y1, x2, y2 = mouth_bbox

        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return frame

        amount = self.STATES.get(
            viseme,
            0.0,
        )

        result = frame.copy()

        if amount <= 0:
            return result

        mouth = result[
            y1:y2,
            x1:x2,
        ]

        mh, mw = mouth.shape[:2]

        if mh < 4 or mw < 4:
            return result

        gray = cv2.cvtColor(
            mouth,
            cv2.COLOR_RGB2GRAY,
        )

        dark_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV
            + cv2.THRESH_OTSU,
        )[1]

        dark_mask = cv2.GaussianBlur(
            dark_mask,
            (5, 5),
            0,
        )

        dark_mask = (
            dark_mask.astype(
                np.float32
            )
            / 255.0
        )

        dark_mask *= amount

        dark_mask = dark_mask[
            ...,
            None
        ]

        output = (
            mouth.astype(
                np.float32
            )
            * (1.0 - 0.45 * dark_mask)
        )

        result[
            y1:y2,
            x1:x2
        ] = np.clip(
            output,
            0,
            255,
        ).astype(
            np.uint8
        )

        return result
