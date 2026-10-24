"""Classical (model-free) text-line detection.

This is a fallback for recognizers that only read single lines. It groups
dark-on-light strokes into horizontal line boxes with morphology. Khmer has
stacked vowels and subscript consonants above and below the baseline, so the
vertical closing kernel is deliberately taller than one used for Latin text.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..preprocessing.io import ImageArray
from ..schemas import BoundingBox


def detect_text_lines(image: ImageArray, *, min_height: int = 8) -> list[BoundingBox]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_w = max(9, w // 60)
    kernel_h = max(3, h // 200)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[BoundingBox] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < min_height or bw < bh or bw * bh < 0.0002 * w * h:
            continue
        if bw > 0.98 * w and bh > 0.5 * h:
            continue
        filled = float(np.count_nonzero(binary[y:y + bh, x:x + bw])) / (bw * bh)
        if filled < 0.08:
            continue
        pad = max(2, bh // 6)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        boxes.append(BoundingBox(x0, y0, x1 - x0, y1 - y0))

    # Reading order: top-to-bottom by line, then left-to-right.
    boxes.sort(key=lambda b: (b.y + b.height // 2) // max(1, min_height * 2) * 10_000 + b.x)
    return boxes
