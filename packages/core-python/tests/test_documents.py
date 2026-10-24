"""Generic template extraction: bilingual lines, document types, Khmer dates."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from khmerocr import KhmerOCR, UnsupportedDocumentError
from khmerocr import jsonschema as contracts
from khmerocr.config import QualityThresholds
from khmerocr.documents import load_builtin_template
from khmerocr.preprocessing import assess_quality
from khmerocr.testing import ScriptedRecognizer, add_glare
from khmerocr.validation import parse_khmer_month_date
from khmerocr.validation.fields import normalize_gender, normalize_issue_date

# Fake values only. Fields sharing a region are recognized once, using the
# hint of the first field in that region.
STUDENT_RESPONSES: dict[str, tuple[str, float | None]] = {
    "full_name_km": ("គំរូ សាកល្បង / KUMRU SAKALBANG", 0.93),
    "gender": ("ស្រី / Female", 0.94),
    "date_of_birth": ("01/02/2003", 0.95),
    "department": ("ដេប៉ាតឺម៉ង់ គំរូ / Department of Samples", 0.9),
    "degree": ("បរិញ្ញាបត្រ / Bachelor", 0.9),
    "address_km": ("ភូមិគំរូ", 0.9),
    "student_id": ("000000002", 0.9),
    "issue_date": ("ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២០", 0.9),
}


def _card() -> np.ndarray:
    """White card with dark text-like bars and a coloured stripe."""
    card = np.full((730, 1152, 3), 255, dtype=np.uint8)
    card[:, :160] = (80, 150, 70)
    rng = np.random.default_rng(0)
    for y in range(140, 330, 28):
        card[y:y + 14, 400:400 + int(rng.integers(150, 700))] = 30
    card[465:490, 200:330] = 40
    return card


def test_student_card_bilingual_fields_are_split() -> None:
    engine = KhmerOCR(ScriptedRecognizer(STUDENT_RESPONSES))
    result = engine.extract_id(_card(), document_type="student_id_card")
    values = {name: f.value for name, f in result.fields.items()}
    assert values["full_name_km"] == "គំរូ សាកល្បង"
    assert values["full_name_latin"] == "KUMRU SAKALBANG"
    assert values["gender"] == "F"
    assert values["date_of_birth"] == "2003-02-01"
    assert values["student_id"] == "000000002"
    assert values["issue_date"] == "2020-01-15"
    assert result.document_type == "student_id_card"
    assert result.review_required and "layout_template_unverified" in result.warnings
    # The shared name region is read once and reported once.
    labels = [r.label for r in result.regions]
    assert labels.count("full_name_km") == 1 and "full_name_latin" not in labels
    contracts.validate(result.to_dict(include_text_regions=True),
                       contracts.contract_for(result.document_type))


def test_missing_separator_does_not_guess_the_language_side() -> None:
    responses = {**STUDENT_RESPONSES, "full_name_km": ("KUMRU SAKALBANG", 0.95)}
    result = KhmerOCR(ScriptedRecognizer(responses)).extract_id(
        _card(), document_type="student_id_card")
    assert result.fields["full_name_km"].value is None
    assert result.fields["full_name_latin"].value is None
    assert "field_not_detected:full_name_latin" in result.warnings


def test_unknown_document_type_is_rejected() -> None:
    with pytest.raises(UnsupportedDocumentError):
        KhmerOCR(ScriptedRecognizer()).extract_id(_card(), document_type="passport")


def test_engine_reports_supported_documents() -> None:
    info = KhmerOCR(ScriptedRecognizer()).info()
    assert set(info["templates"]) == {"khmer_national_id", "student_id_card"}
    assert all(t["verified"] is False for t in info["templates"].values())


def test_student_template_regions_are_inside_the_card() -> None:
    template = load_builtin_template("student_id_card_front_v1")
    for field in template.fields:
        x, y, w, h = field.box
        assert x + w <= 1 and y + h <= 1


def test_khmer_month_dates() -> None:
    assert parse_khmer_month_date("ថ្ងៃទី០១ ខែមីនា ឆ្នាំ២០២៣") == date(2023, 3, 1)
    assert parse_khmer_month_date("ថ្ងៃទី ៣១ ខែ ធ្នូ ឆ្នាំ ១៩៩៩") == date(1999, 12, 31)
    # Lunar-calendar months are not converted.
    assert parse_khmer_month_date("១០កើត ខែផល្គុន ឆ្នាំខាល") is None
    assert parse_khmer_month_date("ថ្ងៃទី៣០ ខែកុម្ភៈ ឆ្នាំ២០២៣") is None


def test_issue_date_normalizer() -> None:
    assert normalize_issue_date("ថ្ងៃទី១៥ ខែមករា ឆ្នាំ២០២០", {}).value == "2020-01-15"
    assert normalize_issue_date("15.01.2020", {}).value == "2020-01-15"
    future = normalize_issue_date("ថ្ងៃទី១ ខែមករា ឆ្នាំ២៩៩៩", {})
    assert not future.valid and "implausible_date" in future.warnings


def test_bilingual_gender_must_agree() -> None:
    assert normalize_gender("ប្រុស / Male", {}).value == "M"
    assert normalize_gender("ស្រី / Female", {}).value == "F"
    conflict = normalize_gender("ប្រុស / Female", {})
    assert conflict.value is None and "ambiguous_value" in conflict.warnings


def test_white_card_background_is_not_glare() -> None:
    thresholds = QualityThresholds()
    clean = assess_quality(_card(), thresholds)
    assert clean.glare_ratio == 0 and clean.usable and "glare_detected" not in clean.issues
    glared = assess_quality(add_glare(_card()[:, :, ::-1].copy() // 2 + 60), thresholds)
    assert "glare_detected" in glared.issues or "severe_glare" in glared.issues
