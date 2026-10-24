"""Document content must never reach logs, even at DEBUG level or on errors."""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card, to_png_bytes
from khmerocr_api.dependencies import Settings
from khmerocr_api.main import create_app

from ..conftest import SENSITIVE_VALUES, SYNTHETIC_RESPONSES


class _FailingRecognizer(ScriptedRecognizer):
    def recognize_line(self, image, hint=None):
        raise ValueError("could not parse 000000001 KUMRU SAKALBANG")


@pytest.fixture
def all_logs(caplog):
    caplog.set_level(logging.DEBUG)
    return caplog


def _assert_clean(caplog) -> None:
    text = "\n".join(r.getMessage() + (r.exc_text or "") for r in caplog.records)
    for value in SENSITIVE_VALUES:
        assert value not in text, f"sensitive value leaked into logs: {value!r}"


def test_successful_extraction_logs_no_content(all_logs) -> None:
    engine = KhmerOCR(ScriptedRecognizer(SYNTHETIC_RESPONSES))
    engine.extract_id(place_on_background(render_card()))
    assert any(r.name == "khmerocr" for r in all_logs.records)  # we do log something
    _assert_clean(all_logs)


def test_failed_extraction_logs_no_content(all_logs) -> None:
    engine = KhmerOCR(_FailingRecognizer())
    with pytest.raises(Exception):
        engine.extract_id(place_on_background(render_card()))
    _assert_clean(all_logs)


def test_api_error_paths_log_no_content(all_logs) -> None:
    png = to_png_bytes(place_on_background(render_card()))
    for recognizer in (ScriptedRecognizer(SYNTHETIC_RESPONSES), _FailingRecognizer()):
        app = create_app(settings=Settings(model="unused"), engine=KhmerOCR(recognizer))
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/v1/documents/national-id/extract",
                        files={"file": ("c.png", png, "image/png")},
                        data={"include_text_regions": "true"})
    _assert_clean(all_logs)
