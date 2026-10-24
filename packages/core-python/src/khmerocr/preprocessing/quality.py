"""Image usability checks: size, blur, glare, exposure, and contrast."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityThresholds
from ..schemas import QualityReport
from .io import ImageArray

_ANALYSIS_LONG_SIDE = 1000


def assess_quality(image: ImageArray, thresholds: QualityThresholds) -> QualityReport:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = _resize_long_side(gray, _ANALYSIS_LONG_SIDE)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.count_nonzero(gray >= 250)) / gray.size
    brightness = float(gray.mean())
    contrast = float(gray.std())

    issues: list[str] = []
    blocking: list[str] = []

    if min(width, height) < thresholds.min_short_side:
        blocking.append("image_too_small")
    if contrast < thresholds.min_contrast:
        blocking.append("low_contrast")
    if brightness < thresholds.min_brightness:
        blocking.append("image_too_dark")
    elif brightness > thresholds.max_brightness:
        blocking.append("image_too_bright")

    if blur_score < thresholds.blur_reject:
        blocking.append("too_blurry")
    elif blur_score < thresholds.blur_warn:
        issues.append("blurry")
    if glare_ratio > thresholds.glare_reject_ratio:
        blocking.append("severe_glare")
    elif glare_ratio > thresholds.glare_warn_ratio:
        issues.append("glare_detected")
    if height > width:
        issues.append("portrait_orientation")

    return QualityReport(
        width=width,
        height=height,
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        brightness=brightness,
        contrast=contrast,
        issues=blocking + issues,
        blocking_issues=blocking,
    )


def _resize_long_side(gray: np.ndarray, long_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return gray
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=interpolation)
