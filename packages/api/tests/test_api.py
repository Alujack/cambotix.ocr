from __future__ import annotations

import io

from fastapi.testclient import TestClient
from khmerocr import KhmerOCR
from khmerocr import jsonschema as contracts
from khmerocr.testing import ScriptedRecognizer
from khmerocr_api.main import create_app
from PIL import Image

from .conftest import GOOD_RESPONSES, SMALL_LIMIT_SETTINGS

EXTRACT = "/v1/documents/national-id/extract"


def _upload(data: bytes, content_type: str = "image/png") -> dict:
    return {"file": ("card.png", data, content_type)}


def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    body = response.json()
    contracts.validate(body, "error")
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == response.headers["x-request-id"]
    return body


def test_health(client: TestClient) -> None:
    body = client.get("/v1/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_models(client: TestClient) -> None:
    body = client.get("/v1/models").json()
    assert body["active"]["recognizer"]["backend"] == "scripted"
    assert isinstance(body["installed"], list)


def test_openapi_lists_v1_endpoints(client: TestClient) -> None:
    paths = client.get("/v1/openapi.json").json()["paths"]
    assert {"/v1/health", "/v1/models", "/v1/ocr", EXTRACT,
            "/v1/documents/quality-check"} <= set(paths)


def test_extract_national_id(client: TestClient, card_png: bytes) -> None:
    response = client.post(EXTRACT, files=_upload(card_png))
    assert response.status_code == 200, response.text
    body = response.json()
    contracts.validate(body, "khmer_national_id")
    assert body["fields"]["id_number"]["value"] == "000000001"
    assert body["review_required"] is True
    assert "text_regions" not in body


def test_extract_with_text_regions(client: TestClient, card_png: bytes) -> None:
    response = client.post(EXTRACT, files=_upload(card_png),
                           data={"include_text_regions": "true"})
    assert len(response.json()["text_regions"]) == len(GOOD_RESPONSES)


def test_quality_check(client: TestClient, card_png: bytes) -> None:
    body = client.post("/v1/documents/quality-check", files=_upload(card_png)).json()
    contracts.validate(body, "quality_report")
    assert body["usable"] is True


def test_general_ocr(client: TestClient, card_png: bytes) -> None:
    response = client.post("/v1/ocr", files=_upload(card_png))
    assert response.status_code == 200
    contracts.validate(response.json(), "general_text")


def test_rejects_unsupported_content_type(client: TestClient, card_png: bytes) -> None:
    _assert_error(client.post(EXTRACT, files=_upload(card_png, "application/pdf")),
                  415, "UNSUPPORTED_MEDIA_TYPE")


def test_verifies_real_format_not_declared_type(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (400, 300)).save(buffer, format="GIF")
    _assert_error(client.post(EXTRACT, files=_upload(buffer.getvalue(), "image/png")),
                  400, "INVALID_IMAGE")


def test_rejects_empty_upload(client: TestClient) -> None:
    _assert_error(client.post(EXTRACT, files=_upload(b"")), 400, "INVALID_IMAGE")


def test_missing_file_is_invalid_request(client: TestClient) -> None:
    body = _assert_error(client.post(EXTRACT), 422, "INVALID_REQUEST")
    assert body["error"]["details"]["fields"] == ["body.file"]


def test_rejects_oversized_upload_by_content_length(card_png: bytes) -> None:
    app = create_app(settings=SMALL_LIMIT_SETTINGS,
                     engine=KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES)))
    with TestClient(app, raise_server_exceptions=False) as client:
        big = card_png + b"\0" * 400_000
        _assert_error(client.post(EXTRACT, files=_upload(big)), 413, "PAYLOAD_TOO_LARGE")


def test_rejects_oversized_streamed_upload(card_png: bytes) -> None:
    app = create_app(settings=SMALL_LIMIT_SETTINGS,
                     engine=KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES)))

    def chunks():
        for _ in range(10):
            yield b"\0" * 50_000

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(EXTRACT, content=chunks(),
                               headers={"content-type": "multipart/form-data; boundary=x"})
        _assert_error(response, 413, "PAYLOAD_TOO_LARGE")


def test_file_part_over_limit_but_under_body_limit(card_png: bytes) -> None:
    app = create_app(settings=SMALL_LIMIT_SETTINGS,
                     engine=KhmerOCR(ScriptedRecognizer(GOOD_RESPONSES)))
    with TestClient(app, raise_server_exceptions=False) as client:
        data = card_png[:8] + b"\0" * (SMALL_LIMIT_SETTINGS.max_upload_bytes + 10)
        _assert_error(client.post(EXTRACT, files=_upload(data)), 413, "PAYLOAD_TOO_LARGE")


def test_low_quality_image_returns_422(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (800, 500), (128, 128, 128)).save(buffer, format="PNG")
    body = _assert_error(client.post(EXTRACT, files=_upload(buffer.getvalue())),
                         422, "IMAGE_QUALITY_TOO_LOW")
    assert "low_contrast" in body["error"]["details"]["issues"]


def test_model_not_available(client_without_model: TestClient, card_png: bytes) -> None:
    health = client_without_model.get("/v1/health").json()
    assert health["model_loaded"] is False and health["model_error"] == "MODEL_NOT_AVAILABLE"
    body = _assert_error(client_without_model.post(EXTRACT, files=_upload(card_png)),
                         503, "MODEL_NOT_AVAILABLE")
    assert "model_dir" not in body["error"].get("details", {})


def test_unknown_route_uses_error_envelope(client: TestClient) -> None:
    _assert_error(client.get("/v1/nope"), 404, "NOT_FOUND")


def test_request_id_is_echoed_when_safe(client: TestClient) -> None:
    assert client.get("/v1/health", headers={"x-request-id": "abc-123"}).headers[
        "x-request-id"] == "abc-123"
    generated = client.get("/v1/health", headers={"x-request-id": "bad id\n"}).headers[
        "x-request-id"]
    assert generated.startswith("req_")


class _Exploding(ScriptedRecognizer):
    def recognize_line(self, image, hint=None):
        raise RuntimeError("secret recognizer detail 000000001")


def test_unexpected_failure_is_opaque(card_png: bytes) -> None:
    app = create_app(settings=SMALL_LIMIT_SETTINGS.__class__(model="unused"),
                     engine=KhmerOCR(_Exploding()))
    with TestClient(app, raise_server_exceptions=False) as client:
        body = _assert_error(client.post(EXTRACT, files=_upload(card_png)),
                             500, "INFERENCE_FAILED")
        assert "secret" not in body["error"]["message"]
