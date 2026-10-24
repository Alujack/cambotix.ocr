"""Consistent error responses.

Every error uses the envelope ``{"error": {"code", "message", "request_id"}}``.
Messages never echo request content.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from khmerocr.errors import ErrorCode, KhmerOCRError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("khmerocr_api")

STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.INVALID_IMAGE: 400,
    ErrorCode.UNSUPPORTED_DOCUMENT: 422,
    ErrorCode.IMAGE_QUALITY_TOO_LOW: 422,
    ErrorCode.MODEL_NOT_AVAILABLE: 503,
    ErrorCode.INFERENCE_FAILED: 500,
}

_CODE_BY_HTTP_STATUS = {
    400: "INVALID_REQUEST",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "INVALID_REQUEST",
}


class ApiError(Exception):
    """An error raised by the HTTP layer itself (not the engine)."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def request_id_of(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_response(status_code: int, code: str, message: str, request_id: str | None,
                   details: dict[str, Any] | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if request_id:
        error["request_id"] = request_id
    if details:
        error["details"] = details
    return JSONResponse({"error": error}, status_code=status_code)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(KhmerOCRError)
    async def _engine_error(request: Request, exc: KhmerOCRError) -> JSONResponse:
        return error_response(STATUS_BY_CODE.get(exc.code, 500), exc.code.value, exc.message,
                              request_id_of(request), _safe_details(exc.details))

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(exc.status_code, exc.code, exc.message, request_id_of(request))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _CODE_BY_HTTP_STATUS.get(exc.status_code, "INVALID_REQUEST")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return error_response(exc.status_code, code, message, request_id_of(request))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report which parameters failed, never their submitted values.
        fields = sorted({".".join(str(p) for p in err.get("loc", ())) for err in exc.errors()})
        return error_response(422, "INVALID_REQUEST", "The request is invalid.",
                              request_id_of(request), {"fields": fields})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled error type=%s request_id=%s", type(exc).__name__,
                     request_id_of(request))
        return error_response(500, "INTERNAL_ERROR", "An unexpected error occurred.",
                              request_id_of(request))


def _safe_details(details: dict[str, Any]) -> dict[str, Any] | None:
    # Filesystem paths are useful on the CLI but should not leak over HTTP.
    safe = {k: v for k, v in details.items() if k not in {"model_dir", "source", "path"}}
    return safe or None
