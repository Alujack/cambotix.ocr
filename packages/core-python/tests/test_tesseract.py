"""Tesseract backend tests. Skipped unless the tesseract binary is installed and
KHMEROCR_TESSDATA points at a directory containing khm/eng traineddata."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from khmerocr.recognition import RegionHint
from khmerocr.testing import render_card

pytestmark = pytest.mark.tesseract

TESSDATA = os.environ.get("KHMEROCR_TESSDATA")
if not (shutil.which("tesseract") and TESSDATA
        and (Path(TESSDATA) / "khm.traineddata").is_file()):
    pytest.skip("tesseract or Khmer language data not available", allow_module_level=True)


@pytest.fixture(scope="module")
def recognizer():
    from khmerocr.recognition.tesseract import TesseractRecognizer

    return TesseractRecognizer(tessdata_dir=Path(TESSDATA))


def test_reads_latin_line(recognizer) -> None:
    card = render_card()
    crop = card[int(0.43 * 638):int(0.53 * 638), int(0.33 * 1012):]
    result = recognizer.recognize_line(crop, RegionHint(script="latin"))
    assert "KUMRU" in result.text
    assert result.confidence is not None and 0 <= result.confidence <= 1


def test_reads_digits_with_whitelist(recognizer) -> None:
    card = render_card()
    crop = card[int(0.19 * 638):int(0.30 * 638), int(0.55 * 1012):]
    hint = RegionHint(script="digits", languages="eng", char_whitelist="0123456789")
    assert recognizer.recognize_line(crop, hint).text.replace(" ", "") == "000000001"


def test_page_recognition_returns_regions(recognizer) -> None:
    regions = recognizer.recognize_page(render_card())
    assert any("SPECIMEN" in r.text for r in regions)


def test_spaced_digits_use_single_word_fallback(recognizer) -> None:
    import numpy as np
    from khmerocr.testing.synthetic import _LATIN_FONT_CANDIDATES, find_font
    from PIL import Image, ImageDraw, ImageFont

    font_path = find_font(_LATIN_FONT_CANDIDATES)
    font = ImageFont.truetype(font_path, 18) if font_path else ImageFont.load_default(18)
    canvas = Image.new("RGB", (160, 30), (235, 235, 235))
    ImageDraw.Draw(canvas).text((8, 5), "000000002", fill=(30, 30, 30), font=font,
                                spacing=4)
    hint = RegionHint(script="digits", languages="eng", char_whitelist="0123456789")
    result = recognizer.recognize_line(np.asarray(canvas), hint)
    assert result.text.replace(" ", "") == "000000002"
