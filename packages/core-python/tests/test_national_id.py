from __future__ import annotations

import dataclasses

import numpy as np
from khmerocr import KhmerOCR
from khmerocr.config import EngineConfig
from khmerocr.documents import load_builtin_template
from khmerocr.recognition import RecognizedText, Recognizer, RegionHint
from khmerocr.testing import ScriptedRecognizer, add_glare, place_on_background, render_card

from .conftest import GOOD_RESPONSES

CRITICAL = ["id_number", "full_name_km", "full_name_latin", "date_of_birth"]


def _verified_template():
    return dataclasses.replace(load_builtin_template("khmer_national_id_front_v1"), verified=True)


def test_fields_are_mapped_and_normalized(scripted_engine, card_photo_png) -> None:
    result = scripted_engine.extract_id(card_photo_png)
    values = {name: field.value for name, field in result.fields.items()}
    assert values == {
        "id_number": "000000001",
        "full_name_km": "គំរូ សាកល្បង",
        "full_name_latin": "KUMRU SAKALBANG",
        "date_of_birth": "1990-01-01",
        "gender": "M",
    }
    assert all(field.valid for field in result.fields.values())
    assert result.metadata["card_detected"] is True


def test_unverified_template_always_requires_review(scripted_engine, card_photo_png) -> None:
    result = scripted_engine.extract_id(card_photo_png)
    assert result.review_required and result.status == "review_required"
    assert "layout_template_unverified" in result.warnings


def test_verified_template_with_confident_fields_succeeds(card_photo_png) -> None:
    engine = KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.status == "success" and not result.review_required
    assert result.warnings == []


def test_missing_critical_field_requires_review(card_photo_png) -> None:
    responses = {**GOOD_RESPONSES, "id_number": ("", None)}
    engine = KhmerOCR(ScriptedRecognizer(responses), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.review_required
    assert "field_not_detected:id_number" in result.warnings
    assert result.fields["id_number"].value is None


def test_low_confidence_field_is_never_silently_accepted(card_photo_png) -> None:
    responses = {**GOOD_RESPONSES, "date_of_birth": ("01.01.1990", 0.4)}
    engine = KhmerOCR(ScriptedRecognizer(responses), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.review_required
    assert "low_confidence:date_of_birth" in result.warnings


def test_missing_confidence_counts_as_low(card_photo_png) -> None:
    responses = {**GOOD_RESPONSES, "full_name_latin": ("KUMRU SAKALBANG", None)}
    engine = KhmerOCR(ScriptedRecognizer(responses), template=_verified_template())
    assert engine.extract_id(card_photo_png).review_required


def test_corrected_id_number_requires_review(card_photo_png) -> None:
    responses = {**GOOD_RESPONSES, "id_number": ("O0000000I", 0.99)}
    engine = KhmerOCR(ScriptedRecognizer(responses), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.fields["id_number"].value == "000000001"
    assert result.review_required and "ocr_corrected:id_number" in result.warnings


def test_invalid_non_critical_value_requires_review(card_photo_png) -> None:
    responses = {**GOOD_RESPONSES, "gender": ("ប្រុស ស្រី", 0.99)}
    engine = KhmerOCR(ScriptedRecognizer(responses), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.fields["gender"].value is None
    assert "ambiguous_value:gender" in result.warnings


def test_nothing_recognized_is_failed(card_photo_png) -> None:
    engine = KhmerOCR(ScriptedRecognizer({}), template=_verified_template())
    result = engine.extract_id(card_photo_png)
    assert result.status == "failed" and result.review_required
    for name in CRITICAL:
        assert f"field_not_detected:{name}" in result.warnings


def test_glare_forces_review(card_photo_png) -> None:
    engine = KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES), template=_verified_template())
    glared = add_glare(place_on_background(render_card()))
    result = engine.extract_id(glared)
    assert result.review_required and "glare_detected" in result.warnings


def test_tight_crop_warns_about_card_boundary() -> None:
    engine = KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES), template=_verified_template())
    result = engine.extract_id(render_card())
    assert "card_boundary_not_detected" in result.warnings and result.review_required


class _UpsideDownAware(Recognizer):
    """Reads a field only if its crop contains dark text pixels. On an
    upside-down card the ID-number region covers empty card background."""

    name = "upside-down-aware"

    def recognize_line(self, image: np.ndarray, hint: RegionHint | None = None) -> RecognizedText:
        assert hint is not None and hint.name is not None
        if image.mean(axis=2).min() > 80:
            return RecognizedText("", None)
        return RecognizedText(*GOOD_RESPONSES[hint.name])


def test_upside_down_card_is_retried_rotated() -> None:
    engine = KhmerOCR(_UpsideDownAware(), template=_verified_template())
    photo = place_on_background(render_card(), angle=180)
    result = engine.extract_id(photo)
    assert result.fields["id_number"].value == "000000001"
    assert result.metadata["quarter_turns"] == 2
    assert "orientation_corrected" in result.warnings


def test_alternate_orientation_can_be_disabled() -> None:
    config = EngineConfig(try_alternate_orientations=False)
    recognizer = ScriptedRecognizer(GOOD_RESPONSES)
    engine = KhmerOCR(recognizer, config=config)
    engine.extract_id(place_on_background(render_card()))
    assert len(recognizer.calls) == len(GOOD_RESPONSES)


def test_text_regions_are_opt_in(scripted_engine, card_photo_png) -> None:
    result = scripted_engine.extract_id(card_photo_png)
    assert "text_regions" not in result.to_dict()
    regions = result.to_dict(include_text_regions=True)["text_regions"]
    assert [r["label"] for r in regions] == list(GOOD_RESPONSES)
