"""Every public result must validate against its published JSON Schema."""

from __future__ import annotations

import jsonschema
import pytest
from khmerocr import BoundingBox, InvalidImageError, KhmerOCR, TextRegion
from khmerocr import jsonschema as contracts
from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card


@pytest.mark.parametrize("include_regions", [False, True])
def test_national_id_result_matches_schema(scripted_engine, card_photo_png,
                                           include_regions: bool) -> None:
    payload = scripted_engine.extract_id(card_photo_png).to_dict(
        include_text_regions=include_regions)
    contracts.validate(payload, "khmer_national_id")


def test_failed_national_id_result_matches_schema(card_photo_png) -> None:
    payload = KhmerOCR(ScriptedRecognizer({})).extract_id(card_photo_png).to_dict()
    contracts.validate(payload, "khmer_national_id")


def test_batch_failure_matches_schema(scripted_engine) -> None:
    payload = scripted_engine.extract_batch([b"junk"])[0].to_dict()
    contracts.validate(payload, "khmer_national_id")


def test_general_text_result_matches_schema() -> None:
    engine = KhmerOCR(ScriptedRecognizer(
        page_regions=[TextRegion("សួស្តី", 0.9, BoundingBox(0, 0, 10, 10))]))
    payload = engine.recognize(place_on_background(render_card())).to_dict()
    contracts.validate(payload, "general_text")


def test_quality_report_matches_schema(scripted_engine, card_photo_png) -> None:
    contracts.validate(scripted_engine.check_image(card_photo_png).to_dict(), "quality_report")


def test_error_matches_schema() -> None:
    contracts.validate(InvalidImageError("bad").to_dict(), "error")


def test_schema_rejects_unknown_field_properties(scripted_engine, card_photo_png) -> None:
    payload = scripted_engine.extract_id(card_photo_png).to_dict()
    payload["fields"]["id_number"]["guess"] = "x"
    with pytest.raises(jsonschema.ValidationError):
        contracts.validate(payload, "khmer_national_id")
