"""Public entry point: the ``KhmerOCR`` engine."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

from ._version import __version__
from .config import EngineConfig
from .documents.layout import DocumentTemplate
from .documents.national_id import DOCUMENT_TYPE, NationalIDExtractor
from .errors import KhmerOCRError, ModelNotAvailableError, UnsupportedDocumentError
from .models import ModelManifest, resolve_model_dir, verify_bundle
from .pipeline import check_quality, decode, run_general_text, run_national_id
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
        self._id_extractor = NationalIDExtractor(recognizer, self.config, template)

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
            "templates": {DOCUMENT_TYPE: {
                "template_id": self._id_extractor.template.template_id,
                "template_version": self._id_extractor.template.template_version,
                "verified": self._id_extractor.template.verified,
            }},
        }

    def recognize(self, image: ImageInput) -> OCRResult:
        """General Khmer/English text recognition."""
        result = run_general_text(image, self.recognizer, self.config)
        result.metadata.update(self._model_metadata())
        return result

    def extract_id(self, image: ImageInput, *, document_type: str = DOCUMENT_TYPE) -> OCRResult:
        """Extract Cambodian National ID fields from the card front."""
        if document_type != DOCUMENT_TYPE:
            raise UnsupportedDocumentError(f"Unsupported document type '{document_type}'.")
        _ = self.recognizer  # raises if the engine has been closed
        result = run_national_id(image, self._id_extractor, self.config)
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
