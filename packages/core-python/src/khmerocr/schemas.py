"""Public result types.

These types are the stable contract of the SDK. Their ``to_dict`` output must
validate against the JSON Schemas shipped in ``khmerocr/jsonschema`` (mirrored
in the repository-level ``schemas/`` directory).

Confidence semantics:

* ``ExtractedField.confidence`` and ``TextRegion.confidence`` are *recognition*
  confidences reported by the recognizer for the text it read. They say nothing
  about whether that text was mapped to the correct field.
* ``ExtractedField.valid`` says whether the normalized value passed the format
  checks for its field type. A valid value can still be wrong.
* ``review_required`` is the only signal that the result may be used without a
  human checking it, and only after thresholds have been calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

Status = Literal["success", "review_required", "failed"]


@dataclass(frozen=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class TextRegion:
    text: str
    confidence: float | None
    bbox: BoundingBox
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": _round(self.confidence),
            "bbox": self.bbox.to_dict(),
            "label": self.label,
        }


@dataclass
class ExtractedField:
    value: str | None
    confidence: float | None
    source: str
    valid: bool | None = None
    # Text as returned by the recognizer, before normalization. Not included in
    # ``to_dict`` output; text regions carry it when explicitly requested.
    raw_text: str | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": _round(self.confidence),
            "source": self.source,
            "valid": self.valid,
        }


@dataclass
class QualityReport:
    width: int
    height: int
    blur_score: float
    glare_ratio: float
    brightness: float
    contrast: float
    issues: list[str]
    blocking_issues: list[str]

    @property
    def usable(self) -> bool:
        return not self.blocking_issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "usable": self.usable,
            "width": self.width,
            "height": self.height,
            "blur_score": round(self.blur_score, 2),
            "glare_ratio": round(self.glare_ratio, 4),
            "brightness": round(self.brightness, 2),
            "contrast": round(self.contrast, 2),
            "issues": list(self.issues),
            "blocking_issues": list(self.blocking_issues),
        }


@dataclass
class OCRResult:
    document_type: str
    fields: dict[str, ExtractedField]
    review_required: bool
    warnings: list[str]
    metadata: dict[str, Any]
    status: Status = "review_required"
    text: str | None = None
    regions: list[TextRegion] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self, *, include_text_regions: bool = False) -> dict[str, Any]:
        """Serialize to the public JSON contract.

        General text results always include ``text`` and ``regions``. Document
        results include ``text_regions`` only when ``include_text_regions`` is
        set, because raw region text duplicates sensitive field content.
        """
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "document_type": self.document_type,
            "status": self.status,
        }
        if self.document_type == "general_text":
            data["text"] = self.text or ""
            data["regions"] = [region.to_dict() for region in self.regions]
        else:
            data["fields"] = {name: f.to_dict() for name, f in self.fields.items()}
            if include_text_regions:
                data["text_regions"] = [region.to_dict() for region in self.regions]
        data["warnings"] = list(self.warnings)
        data["review_required"] = self.review_required
        data["metadata"] = dict(self.metadata)
        return data


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)
