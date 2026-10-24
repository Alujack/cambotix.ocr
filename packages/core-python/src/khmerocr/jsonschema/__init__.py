"""Bundled JSON Schemas for the public result contracts.

These files are the source of truth; the repository-level ``schemas/``
directory holds copies for non-Python consumers and is checked for drift in CI.
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

SCHEMA_FILES = {
    "khmer_national_id": "national-id.schema.json",
    "document": "document-result.schema.json",
    "general_text": "ocr-result.schema.json",
    "quality_report": "quality-report.schema.json",
    "error": "error.schema.json",
}


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by contract name (see ``SCHEMA_FILES``) or file name."""
    filename = SCHEMA_FILES.get(name, name)
    text = (resources.files(__name__) / filename).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


def contract_for(document_type: str) -> str:
    """Schema contract name for a result's ``document_type``."""
    if document_type in ("khmer_national_id", "general_text"):
        return document_type
    return "document"


def validate(instance: dict[str, Any], name: str) -> None:
    """Validate ``instance`` against a bundled schema.

    Requires the optional ``jsonschema`` dependency
    (``pip install "khmerocr[validation]"``). Raises
    ``jsonschema.ValidationError`` on failure.
    """
    import jsonschema

    jsonschema.validate(instance, load_schema(name),
                        cls=jsonschema.Draft202012Validator)
