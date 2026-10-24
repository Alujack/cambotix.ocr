"""Inference must not make network requests."""

from __future__ import annotations

from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card, to_png_bytes
from khmerocr_api.dependencies import Settings
from khmerocr_api.main import create_app

from ..conftest import SYNTHETIC_RESPONSES


def test_engine_runs_without_network(no_network) -> None:
    engine = KhmerOCR(ScriptedRecognizer(SYNTHETIC_RESPONSES))
    photo = place_on_background(render_card(), angle=9)
    engine.check_image(photo)
    engine.extract_id(photo)
    engine.recognize(photo)
    engine.extract_batch([photo, b"junk"])
    assert no_network == []


def test_api_requests_make_no_outbound_connections(no_network) -> None:
    app = create_app(settings=Settings(model="unused"),
                     engine=KhmerOCR(ScriptedRecognizer(SYNTHETIC_RESPONSES)))
    png = to_png_bytes(place_on_background(render_card()))
    # TestClient talks to the app in-process; it does not open sockets.
    with TestClient(app) as client:
        for path in ("/v1/documents/national-id/extract", "/v1/documents/quality-check",
                     "/v1/ocr"):
            assert client.post(path, files={"file": ("c.png", png, "image/png")}).status_code == 200
    assert no_network == []
