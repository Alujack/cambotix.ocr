"""KhmerOCR: local-first Khmer/English OCR and Cambodian National ID extraction."""

import logging

from ._version import __version__
from .config import EngineConfig, QualityThresholds
from .engine import KhmerOCR
from .errors import (
    ErrorCode,
    ImageQualityTooLowError,
    InferenceFailedError,
    InvalidImageError,
    KhmerOCRError,
    ModelNotAvailableError,
    UnsupportedDocumentError,
)
from .schemas import BoundingBox, ExtractedField, OCRResult, QualityReport, TextRegion

# Library logging is silent unless the host application configures it.
logging.getLogger("khmerocr").addHandler(logging.NullHandler())

__all__ = [
    "BoundingBox",
    "EngineConfig",
    "ErrorCode",
    "ExtractedField",
    "ImageQualityTooLowError",
    "InferenceFailedError",
    "InvalidImageError",
    "KhmerOCR",
    "KhmerOCRError",
    "ModelNotAvailableError",
    "OCRResult",
    "QualityReport",
    "QualityThresholds",
    "TextRegion",
    "UnsupportedDocumentError",
    "__version__",
]
