"""Engine configuration.

The numeric thresholds below are starting points, not calibrated values. They
must be re-tuned against the benchmark set (see ``benchmarks/``) before a
release, and the calibrated values recorded alongside the model version.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class QualityThresholds:
    # Images whose shorter side is below this many pixels are rejected.
    min_short_side: int = 300
    # Decoded pixel budget; protects against decompression bombs.
    max_pixels: int = 40_000_000
    # Variance of the Laplacian measured at a 1000px long side.
    blur_warn: float = 120.0
    blur_reject: float = 8.0
    # Fraction of fully saturated (clipped white) pixels.
    glare_warn_ratio: float = 0.015
    glare_reject_ratio: float = 0.12
    # Mean grey level bounds (0-255).
    min_brightness: float = 35.0
    max_brightness: float = 240.0
    # Grey-level standard deviation below which the image is considered blank.
    min_contrast: float = 12.0


@dataclass(frozen=True)
class EngineConfig:
    max_image_bytes: int = 10 * 1024 * 1024
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    # Fields whose recognition confidence falls below this are flagged for review.
    review_confidence_threshold: float = 0.85
    # Raise IMAGE_QUALITY_TOO_LOW for blocking quality issues instead of
    # attempting extraction anyway.
    reject_low_quality: bool = True
    # Retry extraction with the card rotated 180 degrees when the first pass
    # does not produce valid critical fields.
    try_alternate_orientations: bool = True
    # Relative padding added around each template region before recognition.
    region_padding: float = 0.01
