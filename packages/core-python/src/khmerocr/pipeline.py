"""Processing pipelines composed from the engine's building blocks.

Logging here is limited to step names, timings, and counts. Never log images,
recognized text, field values, or anything derived from document content.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from .config import EngineConfig
from .documents.extractor import TemplateExtractor
from .errors import ImageQualityTooLowError, InferenceFailedError, KhmerOCRError
from .preprocessing.io import ImageArray, ImageInput, load_image
from .preprocessing.quality import assess_quality
from .recognition.base import Recognizer
from .schemas import OCRResult, QualityReport

logger = logging.getLogger("khmerocr")

T = TypeVar("T")


def decode(source: ImageInput, config: EngineConfig) -> ImageArray:
    return load_image(source, max_bytes=config.max_image_bytes,
                      max_pixels=config.quality.max_pixels)


def check_quality(image: ImageArray, config: EngineConfig) -> QualityReport:
    return assess_quality(image, config.quality)


def run_document(source: ImageInput, extractor: TemplateExtractor,
                    config: EngineConfig) -> OCRResult:
    started = time.perf_counter()
    image = decode(source, config)
    quality = check_quality(image, config)
    if quality.blocking_issues and config.reject_low_quality:
        raise ImageQualityTooLowError(
            "The document image is not usable for reliable extraction.",
            details={"issues": quality.blocking_issues},
        )
    result = _guarded(lambda: extractor.extract(image, quality))
    result.metadata["quality"] = _quality_summary(quality)
    result.metadata["processing_ms"] = _elapsed_ms(started)
    logger.debug("document extraction type=%s status=%s review=%s fields=%d ms=%d",
                 result.document_type, result.status, result.review_required, len(result.fields),
                 result.metadata["processing_ms"])
    return result


def run_general_text(source: ImageInput, recognizer: Recognizer,
                     config: EngineConfig) -> OCRResult:
    started = time.perf_counter()
    image = decode(source, config)
    quality = check_quality(image, config)
    if quality.blocking_issues and config.reject_low_quality:
        raise ImageQualityTooLowError(
            "The document image is not usable for reliable recognition.",
            details={"issues": quality.blocking_issues},
        )
    regions = _guarded(lambda: recognizer.recognize_page(image))
    confidences = [r.confidence for r in regions if r.confidence is not None]
    low = [r for r in regions
           if r.confidence is None or r.confidence < config.review_confidence_threshold]
    warnings = list(quality.issues)
    if not regions:
        warnings.append("no_text_detected")
    if low:
        warnings.append("low_confidence_regions")
    review = bool(low) or not regions or bool(quality.issues)
    result = OCRResult(
        document_type="general_text",
        fields={},
        review_required=review,
        warnings=warnings,
        status="failed" if not regions else ("review_required" if review else "success"),
        text="\n".join(r.text for r in regions),
        regions=regions,
        metadata={
            "quality": _quality_summary(quality),
            "region_count": len(regions),
            "mean_confidence": (round(sum(confidences) / len(confidences), 4)
                                if confidences else None),
            "processing_ms": _elapsed_ms(started),
        },
    )
    logger.debug("general_text recognition regions=%d ms=%d", len(regions),
                 result.metadata["processing_ms"])
    return result


def _guarded(step: Callable[[], T]) -> T:
    try:
        return step()
    except KhmerOCRError:
        raise
    except Exception as exc:
        # Do not include the exception message: it may echo document content.
        logger.error("inference step failed: %s", type(exc).__name__)
        raise InferenceFailedError("Document processing failed.") from exc


def _quality_summary(quality: QualityReport) -> dict[str, object]:
    return {
        "usable": quality.usable,
        "issues": list(quality.issues),
        "blur_score": round(quality.blur_score, 2),
        "glare_ratio": round(quality.glare_ratio, 4),
    }


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
