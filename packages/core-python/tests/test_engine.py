from __future__ import annotations

import pytest
from khmerocr import (
    BoundingBox,
    ImageQualityTooLowError,
    KhmerOCR,
    ModelNotAvailableError,
    TextRegion,
    UnsupportedDocumentError,
)
from khmerocr.testing import ScriptedRecognizer, add_blur, place_on_background, render_card

from .conftest import GOOD_RESPONSES


def test_recognize_general_text() -> None:
    regions = [TextRegion("សួស្តី", 0.95, BoundingBox(10, 10, 100, 30)),
               TextRegion("Hello", 0.6, BoundingBox(10, 50, 100, 30))]
    engine = KhmerOCR(ScriptedRecognizer(page_regions=regions))
    result = engine.recognize(place_on_background(render_card()))
    assert result.document_type == "general_text"
    assert result.text == "សួស្តី\nHello"
    assert result.review_required and "low_confidence_regions" in result.warnings


def test_check_image(scripted_engine, card_photo_png) -> None:
    report = scripted_engine.check_image(card_photo_png)
    assert report.usable and report.width == 1600


def test_unusable_image_raises_quality_error(scripted_engine) -> None:
    blurred = add_blur(place_on_background(render_card()), 12)
    with pytest.raises(ImageQualityTooLowError) as excinfo:
        scripted_engine.extract_id(blurred)
    assert "too_blurry" in excinfo.value.details["issues"]


def test_unsupported_document_type(scripted_engine, card_photo_png) -> None:
    with pytest.raises(UnsupportedDocumentError):
        scripted_engine.extract_id(card_photo_png, document_type="passport")


def test_batch_isolates_failures(scripted_engine, card_photo_png) -> None:
    results = scripted_engine.extract_batch([card_photo_png, b"not an image"])
    assert results[0].fields["id_number"].value == "000000001"
    assert results[1].status == "failed"
    assert results[1].metadata["error"]["code"] == "INVALID_IMAGE"


def test_close_releases_recognizer(card_photo_png) -> None:
    recognizer = ScriptedRecognizer(GOOD_RESPONSES)
    with KhmerOCR(recognizer) as engine:
        engine.extract_id(card_photo_png)
    assert recognizer.closed
    with pytest.raises(ModelNotAvailableError):
        engine.extract_id(card_photo_png)
    engine.close()  # idempotent
