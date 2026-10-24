"""Public entry point: the ``KhmerOCR`` engine."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

from ._version import __version__
from .config import EngineConfig
from .documents.extractor import TemplateExtractor
from .documents.layout import DOCUMENT_TEMPLATES, DocumentTemplate, template_for_document
from .documents.national_id import DOCUMENT_TYPE
from .errors import KhmerOCRError, ModelNotAvailableError
from .models import ModelManifest, resolve_model_dir, verify_bundle
from .pipeline import check_quality, decode, run_document, run_general_text
from .preprocessing.io import ImageInput
from .recognition.base import Recognizer
from .recognition.registry import create_recognizer
from .schemas import OCRResult, QualityReport

logger = logging.getLogger("khmerocr")


class KhmerOCR:
    """Local Khmer/English OCR engine.

    The engine runs entirely offline. It never downloads models, calls remote
    services, stores images, or logs document content. Instances are not
    thread-safe; use one instance per worker or serialize calls.
    """

    def __init__(self, recognizer: Recognizer, *, config: EngineConfig | None = None,
                 manifest: ModelManifest | None = None,
                 template: DocumentTemplate | None = None) -> None:
        self.config = config or EngineConfig()
        self.manifest = manifest
        self._recognizer: Recognizer | None = recognizer
        # Optional template override; it replaces the default template for its
        # own document type.
        self._template_overrides = {template.document_type: template} if template else {}
        self._extractors: dict[str, TemplateExtractor] = {}

    @classmethod
    def from_pretrained(cls, model_dir: str | os.PathLike[str], *,
                        config: EngineConfig | None = None,
                        template: DocumentTemplate | None = None) -> KhmerOCR:
        """Load an installed model bundle by directory path or model ID.

        Every bundle file is checked against the manifest checksums before
        the recognizer is created.
        """
        path = resolve_model_dir(model_dir)
        manifest = ModelManifest.load(path)
        verify_bundle(path, manifest)
        if not manifest.license_approved:
            logger.warning("model %s has no approved license record; do not redistribute it",
                           manifest.model_id)
        recognizer = create_recognizer(manifest.recognizer_backend, path.resolve(),
                                       manifest.recognizer_options)
        return cls(recognizer, config=config, manifest=manifest, template=template)

    @property
    def recognizer(self) -> Recognizer:
        if self._recognizer is None:
            raise ModelNotAvailableError("The engine has been closed.")
        return self._recognizer

    def info(self) -> dict[str, Any]:
        return {
            "sdk_version": __version__,
            "model": self.manifest.summary() if self.manifest else None,
            "recognizer": self.recognizer.describe(),
            "templates": {
                document_type: {
                    "template_id": extractor.template.template_id,
                    "template_version": extractor.template.template_version,
                    "verified": extractor.template.verified,
                }
                for document_type, extractor in (
                    (dt, self._extractor(dt)) for dt in self.supported_documents())
            },
        }

    def recognize(self, image: ImageInput) -> OCRResult:
        """General Khmer/English text recognition."""
        result = run_general_text(image, self.recognizer, self.config)
        result.metadata.update(self._model_metadata())
        return result

    @staticmethod
    def supported_documents() -> list[str]:
        return sorted(DOCUMENT_TEMPLATES)

    def extract_id(self, image: ImageInput, *, document_type: str = DOCUMENT_TYPE) -> OCRResult:
        """Extract fields from the front of an ID card.

        ``document_type`` defaults to the Cambodian National ID; see
        ``supported_documents()`` for the others.
        """
        extractor = self._extractor(document_type)
        result = run_document(image, extractor, self.config)
        result.metadata.update(self._model_metadata())
        return result

    def check_image(self, image: ImageInput) -> QualityReport:
        """Report blur, glare, exposure, size, and orientation problems."""
        return check_quality(decode(image, self.config), self.config)

    def extract_batch(self, images: Iterable[ImageInput], *,
                      document_type: str = DOCUMENT_TYPE) -> list[OCRResult]:
        """Extract fields from several documents.

        A document that fails produces a ``failed`` result carrying the error
        code as a warning, so one bad image does not abort the batch.
        """
        results: list[OCRResult] = []
        for image in images:
            try:
                results.append(self.extract_id(image, document_type=document_type))
            except ModelNotAvailableError:
                raise
            except KhmerOCRError as exc:
                results.append(OCRResult(
                    document_type=document_type, fields={}, review_required=True,
                    warnings=[exc.code.value.lower()], status="failed",
                    metadata={"error": {"code": exc.code.value, "message": exc.message},
                              **self._model_metadata()},
                ))
        return results

    def _extractor(self, document_type: str) -> TemplateExtractor:
        extractor = self._extractors.get(document_type)
        if extractor is None:
            template = (self._template_overrides.get(document_type)
                        or template_for_document(document_type))
            extractor = TemplateExtractor(self.recognizer, self.config, template)
            self._extractors[document_type] = extractor
        _ = self.recognizer  # raises if the engine has been closed
        return extractor

    def close(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None

    def __enter__(self) -> KhmerOCR:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _model_metadata(self) -> dict[str, Any]:
        recognizer = self.recognizer.describe()
        data: dict[str, Any] = {"sdk_version": __version__,
                                "recognizer_backend": recognizer.get("backend")}
        if self.manifest:
            data["model_id"] = self.manifest.model_id
            data["model_version"] = self.manifest.model_version
        return data
