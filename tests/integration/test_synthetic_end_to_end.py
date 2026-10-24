"""Synthetic photos through the full stack: image -> engine -> REST -> schema."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr import jsonschema as contracts
from khmerocr.testing import (
    ScriptedRecognizer,
    add_blur,
    place_on_background,
    render_card,
    to_png_bytes,
)
from khmerocr_api.dependencies import Settings
from khmerocr_api.main import create_app

from ..conftest import SYNTHETIC_RESPONSES


@pytest.mark.parametrize("angle,perspective,blur", [
    (0, 0.0, 0.0), (15, 0.03, 0.0), (-25, 0.05, 0.0), (5, 0.0, 1.5), (180, 0.02, 0.0),
])
def test_varied_captures(angle: float, perspective: float, blur: float) -> None:
    photo = place_on_background(render_card(), angle=angle, perspective=perspective, seed=11)
    if blur:
        photo = add_blur(photo, blur)
    app = create_app(settings=Settings(model="unused"),
                     engine=KhmerOCR(ScriptedRecognizer(SYNTHETIC_RESPONSES)))
    with TestClient(app) as client:
        response = client.post("/v1/documents/national-id/extract",
                               files={"file": ("c.png", to_png_bytes(photo), "image/png")})
    assert response.status_code == 200, response.text
    body = response.json()
    contracts.validate(body, "khmer_national_id")
    assert body["metadata"]["card_detected"] is True
    assert body["review_required"] is True  # provisional template
    if blur:
        assert "blurry" in body["warnings"]
