from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr.testing import ScriptedRecognizer, place_on_background, render_card, to_png_bytes
from khmerocr_api.dependencies import Settings
from khmerocr_api.main import create_app

GOOD_RESPONSES: dict[str, tuple[str, float | None]] = {
    "id_number": ("000000001", 0.97),
    "full_name_km": ("គោត្តនាមនិងនាម: គំរូ សាកល្បង", 0.93),
    "full_name_latin": ("KUMRU SAKALBANG", 0.95),
    "date_of_birth": ("ថ្ងៃខែឆ្នាំកំណើត: 01.01.1990", 0.91),
    "gender": ("ភេទ: ប្រុស", 0.9),
}

SMALL_LIMIT_SETTINGS = Settings(model="unused", max_upload_bytes=200_000)


@pytest.fixture(scope="session")
def card_png() -> bytes:
    return to_png_bytes(place_on_background(render_card(), angle=5, seed=2))


@pytest.fixture
def client() -> Iterator[TestClient]:
    engine = KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES))
    app = create_app(settings=Settings(model="unused"), engine=engine)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path, monkeypatch) -> Iterator[TestClient]:
    monkeypatch.setenv("KHMEROCR_HOME", str(tmp_path))
    app = create_app(settings=Settings(model="khmerocr-id-v1"))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client
