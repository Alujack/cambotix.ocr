"""Template-driven field extraction for card documents.

Fields are mapped deterministically: the card is rectified to a canonical size
and each field is read from a region defined by a versioned layout template.
No language model is used to guess or complete identity fields.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..config import EngineConfig
from ..errors import UnsupportedDocumentError
from ..preprocessing.geometry import normalize_card, rotate_quarter_turns
from ..preprocessing.io import ImageArray
from ..recognition.base import RecognizedText, Recognizer
from ..schemas import BoundingBox, ExtractedField, OCRResult, QualityReport, Status, TextRegion
from ..validation.fields import NORMALIZERS, Normalized
from ..validation.text import normalize_text, strip_labels
from .layout import DocumentTemplate, FieldSpec

# Quality issues that make field positions or pixels unreliable.
_REVIEW_QUALITY_ISSUES = {"blurry", "glare_detected"}


@dataclass
class _Pass:
    quarter_turns: int
    fields: dict[str, ExtractedField] = field(default_factory=dict)
    regions: list[TextRegion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_review: set[str] = field(default_factory=set)

    def score(self, critical: list[str]) -> tuple[int, float]:
        valid = [self.fields[name] for name in critical
                 if name in self.fields and self.fields[name].valid]
        confidence = sum(f.confidence or 0.0 for f in valid) / max(1, len(critical))
        return len(valid), confidence


class TemplateExtractor:
    def __init__(self, recognizer: Recognizer, config: EngineConfig,
                 template: DocumentTemplate) -> None:
        self.recognizer = recognizer
        self.config = config
        self.template = template
        unknown = sorted({f.type for f in template.fields} - set(NORMALIZERS))
        if unknown:
            raise UnsupportedDocumentError("The template uses unknown field types.",
                                           details={"field_types": unknown})

    @property
    def document_type(self) -> str:
        return self.template.document_type

    def extract(self, image: ImageArray, quality: QualityReport) -> OCRResult:
        template = self.template
        critical = template.critical_fields
        card = normalize_card(image, template.canonical_size)

        turns_to_try = [0, 2] if self.config.try_alternate_orientations else [0]
        best: _Pass | None = None
        for extra_turns in turns_to_try:
            attempt = self._read_fields(rotate_quarter_turns(card.image, extra_turns),
                                        (card.quarter_turns + extra_turns) % 4)
            if best is None or attempt.score(critical) > best.score(critical):
                best = attempt
            # Any field needing review triggers the 180-degree pass. Cheaper
            # heuristics (e.g. "half the critical fields are valid") fail
            # because upside-down garbage can pass weak format checks.
            if not attempt.needs_review:
                break
        assert best is not None

        warnings: list[str] = []
        review = bool(best.needs_review)
        quality_flags = [issue for issue in quality.issues if issue in _REVIEW_QUALITY_ISSUES]
        if quality_flags:
            warnings.extend(quality_flags)
            review = True
        if not card.detected:
            warnings.append("card_boundary_not_detected")
            review = True
        if best.quarter_turns:
            warnings.append("orientation_corrected")
        if not template.verified:
            warnings.append("layout_template_unverified")
            review = True
        warnings.extend(best.warnings)

        status: Status
        if all(f.value is None for f in best.fields.values()):
            status = "failed"
            review = True
        else:
            status = "review_required" if review else "success"

        return OCRResult(
            document_type=template.document_type,
            fields=best.fields,
            review_required=review,
            warnings=_dedupe(warnings),
            status=status,
            regions=best.regions,
            metadata={
                "template_id": template.template_id,
                "template_version": template.template_version,
                "template_verified": template.verified,
                "card_detected": card.detected,
                "quarter_turns": best.quarter_turns,
                "critical_fields": critical,
                "review_confidence_threshold": self.config.review_confidence_threshold,
                "coordinate_space": {"width": template.canonical_size[0],
                                     "height": template.canonical_size[1]},
            },
        )

    def _read_fields(self, card: ImageArray, quarter_turns: int) -> _Pass:
        result = _Pass(quarter_turns=quarter_turns)
        padding = (self.template.padding if self.template.padding is not None
                   else self.config.region_padding)
        # Fields that share a region (e.g. "Khmer / English" on one line) reuse
        # the same reads.
        reads_by_region: dict[tuple[object, ...], _RegionReads] = {}
        for spec in self.template.fields:
            key = (spec.box, spec.script, spec.languages, spec.char_whitelist)
            bbox = _pixel_box(spec, card.shape[1], card.shape[0], padding)
            reads = reads_by_region.get(key)
            first_use = reads is None
            if reads is None:
                crop = card[bbox.y:bbox.y + bbox.height, bbox.x:bbox.x + bbox.width]
                reads = _RegionReads(self.recognizer.recognize_candidates(crop, spec.hint()))
                reads_by_region[key] = reads
            field, recognized = self._map_field(spec, reads, result)
            result.fields[spec.name] = field
            if first_use:
                result.regions.append(TextRegion(text=normalize_text(recognized.text),
                                                 confidence=recognized.confidence,
                                                 bbox=bbox, label=spec.name))
        return result

    def _map_field(self, spec: FieldSpec, reads: _RegionReads,
                   result: _Pass) -> tuple[ExtractedField, RecognizedText]:
        """Map the first candidate read whose value validates; if none does,
        fall back to the primary read so its problems are reported."""
        name = spec.name
        chosen: tuple[int, RecognizedText, str, Normalized | None] | None = None
        primary: tuple[int, RecognizedText, str, Normalized | None] | None = None
        index = 0
        while (recognized := reads.get(index)) is not None:
            raw = _select_part(normalize_text(recognized.text), spec)
            normalized = (NORMALIZERS[spec.type](strip_labels(raw, list(spec.labels)),
                                                 spec.options()) if raw else None)
            if primary is None:
                primary = (index, recognized, raw, normalized)
            if normalized is not None and normalized.valid:
                chosen = (index, recognized, raw, normalized)
                break
            index += 1
        assert primary is not None
        used, recognized, raw, maybe_normalized = chosen or primary
        if used:
            result.warnings.append(f"fallback_recognizer:{name}")
        confidence = recognized.confidence
        if not raw or maybe_normalized is None:
            result.warnings.append(f"field_not_detected:{name}")
            if spec.critical:
                result.needs_review.add(name)
            return ExtractedField(None, None, spec.source, valid=False), recognized

        normalized = maybe_normalized
        result.warnings.extend(f"{code}:{name}" for code in normalized.warnings)
        if normalized.value is None:
            result.warnings.append(f"field_unreadable:{name}")
        if confidence is None or confidence < self.config.review_confidence_threshold:
            result.warnings.append(f"low_confidence:{name}")
            low_confidence = True
        else:
            low_confidence = False

        if normalized.value is None:
            if spec.critical:
                result.needs_review.add(name)
        elif not normalized.valid or low_confidence or "ocr_corrected" in normalized.warnings:
            result.needs_review.add(name)

        return ExtractedField(normalized.value, confidence, spec.source,
                              valid=normalized.valid, raw_text=raw), recognized


class _RegionReads:
    """Lazily evaluated candidate reads of one region, shared between fields."""

    def __init__(self, candidates: Iterator[RecognizedText]) -> None:
        self._candidates = candidates
        self._reads: list[RecognizedText] = []

    def get(self, index: int) -> RecognizedText | None:
        while len(self._reads) <= index:
            next_read = next(self._candidates, None)
            if next_read is None:
                return None
            self._reads.append(next_read)
        return self._reads[index]


def _select_part(text: str, spec: FieldSpec) -> str:
    """Pick one side of a bilingual line such as ``"ប្រុស / Male"``."""
    if spec.separator is None or spec.part is None:
        return text
    parts = text.split(spec.separator)
    if len(parts) < 2:
        # Separator not recognized: return nothing rather than guess which
        # side the text belongs to.
        return ""
    if spec.part >= len(parts):
        return ""
    return parts[spec.part].strip()


def _pixel_box(spec: FieldSpec, width: int, height: int, padding: float) -> BoundingBox:
    x, y, w, h = spec.box
    x0 = max(0, int((x - padding) * width))
    y0 = max(0, int((y - padding) * height))
    x1 = min(width, int((x + w + padding) * width))
    y1 = min(height, int((y + h + padding) * height))
    return BoundingBox(x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))
