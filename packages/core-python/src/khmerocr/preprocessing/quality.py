"""Image usability checks: size, blur, glare, exposure, and contrast."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import QualityThresholds
from ..schemas import QualityReport
from .io import ImageArray

_ANALYSIS_LONG_SIDE = 1000
_SATURATED = 250
# Glare hotspots fade out through a soft bright halo; printed white card
# backgrounds end at sharp edges (text, photo, borders). A saturated region
# counts as glare when this share of its surrounding ring is bright-but-not-
# clipped. Separation measured on synthetic glare (>=0.80) versus a real
# white-background card (0.32); recalibrate on real glare photos.
_HALO_RING_PX = 10
_HALO_MIN_FRACTION = 0.6
_MIN_GLARE_COMPONENT_PX = 50


def assess_quality(image: ImageArray, thresholds: QualityThresholds) -> QualityReport:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = _resize_long_side(gray, _ANALYSIS_LONG_SIDE)

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = _glare_ratio(gray)
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


def _glare_ratio(gray: np.ndarray) -> float:
    """Share of the image covered by saturated regions that look like glare."""
    saturated = (gray >= _SATURATED).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(saturated, connectivity=8)
    kernel = np.ones((2 * _HALO_RING_PX + 1, 2 * _HALO_RING_PX + 1), np.uint8)
    h, w = gray.shape
    glare_pixels = 0
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        if area < _MIN_GLARE_COMPONENT_PX:
            continue
        x0, y0 = max(0, x - _HALO_RING_PX), max(0, y - _HALO_RING_PX)
        x1, y1 = min(w, x + bw + _HALO_RING_PX), min(h, y + bh + _HALO_RING_PX)
        component = (labels[y0:y1, x0:x1] == label).astype(np.uint8)
        ring = cv2.dilate(component, kernel) > component
        ring_values = gray[y0:y1, x0:x1][ring]
        if ring_values.size == 0:
            continue
        halo = np.count_nonzero((ring_values >= 200) & (ring_values < _SATURATED))
        if halo / ring_values.size >= _HALO_MIN_FRACTION:
            glare_pixels += int(area)
    return glare_pixels / gray.size


def _resize_long_side(gray: np.ndarray, long_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = long_side / max(h, w)
    if abs(scale - 1.0) < 1e-3:
        return gray
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=interpolation)
