"""Recognizer interface.

Every OCR backend (Tesseract, Netra-OCR, PaddleOCR, ...) is wrapped in a
``Recognizer`` so the rest of the engine never depends on a specific model.
Recognizers must run fully offline: they may only read files from the model
bundle directory and must never download weights or call remote services.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..detection import detect_text_lines
from ..preprocessing.io import ImageArray
from ..schemas import TextRegion

Script = Literal["khmer", "latin", "digits", "mixed"]


@dataclass(frozen=True)
class RegionHint:
    """What the engine expects to find in a crop.

    Backends may use hints to select languages or character sets. A backend
    must still return what it actually reads, not what the hint suggests.
    """

    name: str | None = None
    script: Script = "mixed"
    languages: str | None = None
    char_whitelist: str | None = None


@dataclass(frozen=True)
class RecognizedText:
    text: str
    # Recognition confidence in [0, 1], or None if the backend has no score.
    confidence: float | None


class Recognizer(ABC):
    name: str = "recognizer"

    @abstractmethod
    def recognize_line(self, image: ImageArray, hint: RegionHint | None = None) -> RecognizedText:
        """Recognize a single text line or field crop."""

    def recognize_page(self, image: ImageArray) -> list[TextRegion]:
        """Recognize all text on a page. Backends with native layout analysis
        should override this; the default detects lines and reads each one."""
        regions: list[TextRegion] = []
        for box in detect_text_lines(image):
            crop = image[box.y:box.y + box.height, box.x:box.x + box.width]
            result = self.recognize_line(crop)
            if result.text.strip():
                regions.append(TextRegion(text=result.text, confidence=result.confidence, bbox=box))
        return regions

    def describe(self) -> dict[str, str]:
        return {"backend": self.name}

    def close(self) -> None:
        """Release backend resources. Must be idempotent."""
        return None
