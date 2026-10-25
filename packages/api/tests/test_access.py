"""API key and CORS settings for clients on other machines (apps, browsers)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr import jsonschema as contracts
from khmerocr.testing import ScriptedRecognizer
from khmerocr_api.dependencies import Settings
from khmerocr_api.main import create_app

from .conftest import GOOD_RESPONSES

EXTRACT = "/v1/documents/national-id/extract"
KEY = "test-key-0123456789abcdef"
ORIGIN = "https://app.example.com"


@pytest.fixture
def secured() -> Iterator[TestClient]:
    app = create_app(settings=Settings(model="unused", api_key=KEY, cors_origins=(ORIGIN,)),
                     engine=KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES)))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _upload(data: bytes) -> dict:
    return {"file": ("card.png", data, "image/png")}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-key-0123456789"},
                                     {"X-API-Key": "wrong"}, {"Authorization": KEY}])
def test_missing_or_wrong_key_is_rejected(secured: TestClient, card_png: bytes,
                                          headers: dict) -> None:
    response = secured.post(EXTRACT, files=_upload(card_png), headers=headers)
    assert response.status_code == 401
    body = response.json()
    contracts.validate(body, "error")
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("headers", [{"Authorization": f"Bearer {KEY}"},
                                     {"authorization": f"bearer {KEY}"}, {"X-API-Key": KEY}])
def test_valid_key_is_accepted(secured: TestClient, card_png: bytes, headers: dict) -> None:
    response = secured.post(EXTRACT, files=_upload(card_png), headers=headers)
    assert response.status_code == 200, response.text


def test_health_needs_no_key(secured: TestClient) -> None:
    assert secured.get("/v1/health").status_code == 200


def test_every_other_endpoint_needs_the_key(secured: TestClient) -> None:
    for path in ("/v1/models", "/v1/openapi.json", "/v1/docs"):
        assert secured.get(path).status_code == 401, path


def test_cors_preflight_is_answered_without_a_key(secured: TestClient) -> None:
    response = secured.options(EXTRACT, headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ORIGIN


def test_cors_headers_reach_browsers_on_errors_too(secured: TestClient,
                                                   card_png: bytes) -> None:
    response = secured.post(EXTRACT, files=_upload(card_png), headers={"Origin": ORIGIN})
    assert response.status_code == 401
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert "x-request-id" in response.headers["access-control-expose-headers"].lower()


def test_other_origins_get_no_cors_headers(secured: TestClient) -> None:
    response = secured.get("/v1/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


def test_no_cors_headers_by_default(client: TestClient) -> None:
    response = client.get("/v1/health", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_short_api_key_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        Settings(model="unused", api_key="short")


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("KHMEROCR_API_KEY", KEY)
    monkeypatch.setenv("KHMEROCR_CORS_ORIGINS", f" {ORIGIN} , http://localhost:5173,")
    settings = Settings.from_env()
    assert settings.api_key == KEY
    assert settings.cors_origins == (ORIGIN, "http://localhost:5173")
