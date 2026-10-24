"""Deterministic recognizers for tests, examples, and pipeline development."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..preprocessing.io import ImageArray
from ..recognition.base import RecognizedText, Recognizer, RegionHint
from ..schemas import TextRegion


class ScriptedRecognizer(Recognizer):
    """Returns pre-set text per field name (taken from ``RegionHint.name``).

    Useful for exercising field mapping, normalization, and review logic
    without a real OCR model.
    """

    name = "scripted"

    def __init__(self, responses: Mapping[str, tuple[str, float | None]] | None = None,
                 *, page_regions: list[TextRegion] | None = None) -> None:
        self.responses = dict(responses or {})
        self.page_regions = list(page_regions or [])
        self.calls: list[str | None] = []
        self.closed = False

    def recognize_line(self, image: ImageArray, hint: RegionHint | None = None) -> RecognizedText:
        name = hint.name if hint else None
        self.calls.append(name)
        text, confidence = self.responses.get(name or "", ("", None))
        return RecognizedText(text=text, confidence=confidence)

    def recognize_page(self, image: ImageArray) -> list[TextRegion]:
        return list(self.page_regions)

    def close(self) -> None:
        self.closed = True


def scripted_factory(model_dir: Path, options: Mapping[str, Any]) -> Recognizer:
    """Registry factory: reads responses from the manifest's recognizer options."""
    responses = {name: (str(value[0]), value[1])
                 for name, value in dict(options.get("responses", {})).items()}
    return ScriptedRecognizer(responses)
