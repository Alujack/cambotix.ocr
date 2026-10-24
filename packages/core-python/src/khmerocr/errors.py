"""Stable error types.

Error codes are part of the public contract (they are returned verbatim by the
REST API), so never rename an existing code. Error messages must never contain
document content such as names, ID numbers, or recognized text.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_IMAGE = "INVALID_IMAGE"
    UNSUPPORTED_DOCUMENT = "UNSUPPORTED_DOCUMENT"
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    IMAGE_QUALITY_TOO_LOW = "IMAGE_QUALITY_TOO_LOW"
    INFERENCE_FAILED = "INFERENCE_FAILED"


class KhmerOCRError(Exception):
    """Base class for all errors raised by the engine."""

    code: ErrorCode = ErrorCode.INFERENCE_FAILED

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"error": error}


class InvalidImageError(KhmerOCRError):
    code = ErrorCode.INVALID_IMAGE


class UnsupportedDocumentError(KhmerOCRError):
    code = ErrorCode.UNSUPPORTED_DOCUMENT


class ModelNotAvailableError(KhmerOCRError):
    code = ErrorCode.MODEL_NOT_AVAILABLE


class ImageQualityTooLowError(KhmerOCRError):
    code = ErrorCode.IMAGE_QUALITY_TOO_LOW


class InferenceFailedError(KhmerOCRError):
    code = ErrorCode.INFERENCE_FAILED
