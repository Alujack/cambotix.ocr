"""Cambodian National ID field extraction."""

from __future__ import annotations

from ..config import EngineConfig
from ..errors import UnsupportedDocumentError
from ..recognition.base import Recognizer
from .extractor import TemplateExtractor
from .layout import DocumentTemplate, load_builtin_template

DOCUMENT_TYPE = "khmer_national_id"
DEFAULT_TEMPLATE_ID = "khmer_national_id_front_v1"


class NationalIDExtractor(TemplateExtractor):
    def __init__(self, recognizer: Recognizer, config: EngineConfig,
                 template: DocumentTemplate | None = None) -> None:
        template = template or load_builtin_template(DEFAULT_TEMPLATE_ID)
        if template.document_type != DOCUMENT_TYPE:
            raise UnsupportedDocumentError("The template is not a National ID template.")
        super().__init__(recognizer, config, template)
