"""Settings, engine lifecycle, and request helpers."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anyio
from fastapi import Request, UploadFile
from khmerocr import KhmerOCR, KhmerOCRError, ModelNotAvailableError
from khmerocr import jsonschema as contracts

from .errors import ApiError

logger = logging.getLogger("khmerocr_api")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MIN_API_KEY_LENGTH = 16
T = TypeVar("T")


@dataclass(frozen=True)
class Settings:
    model: str = "khmerocr-id-v1"
    max_upload_bytes: int = 10 * 1024 * 1024
    # Concurrent inference calls. The engine is not thread-safe, so values
    # above 1 require a recognizer backend documented as thread-safe.
    max_concurrency: int = 1
    validate_responses: bool = True
    # When set, every endpoint except /v1/health requires this key. Required
    # before the service is reachable from other machines (phones, browsers).
    api_key: str | None = None
    # Browser origins allowed to call the API (CORS). Empty: no CORS headers.
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.api_key is not None and len(self.api_key) < MIN_API_KEY_LENGTH:
            raise ValueError(f"KHMEROCR_API_KEY must be at least {MIN_API_KEY_LENGTH} "
                             "characters.")

    @classmethod
    def from_env(cls) -> Settings:
        origins = os.environ.get("KHMEROCR_CORS_ORIGINS", "")
        return cls(
            model=os.environ.get("KHMEROCR_MODEL", cls.model),
            max_upload_bytes=int(os.environ.get("KHMEROCR_MAX_UPLOAD_BYTES",
                                                cls.max_upload_bytes)),
            max_concurrency=int(os.environ.get("KHMEROCR_MAX_CONCURRENCY", cls.max_concurrency)),
            validate_responses=os.environ.get("KHMEROCR_VALIDATE_RESPONSES", "1") != "0",
            api_key=os.environ.get("KHMEROCR_API_KEY") or None,
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
        )


@dataclass
class EngineState:
    settings: Settings
    engine: KhmerOCR | None = None
    load_error: KhmerOCRError | None = None
    limiter: anyio.CapacityLimiter = field(init=False)

    def __post_init__(self) -> None:
        self.limiter = anyio.CapacityLimiter(self.settings.max_concurrency)

    def load(self) -> None:
        try:
            self.engine = KhmerOCR.from_pretrained(self.settings.model)
        except KhmerOCRError as exc:
            # The service still starts so /health can report the problem.
            self.load_error = exc
            logger.error("model could not be loaded: %s", exc.code.value)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def get_state(request: Request) -> EngineState:
    state: EngineState = request.app.state.engine_state
    return state


def get_engine(request: Request) -> KhmerOCR:
    state = get_state(request)
    if state.engine is None:
        if state.load_error is not None:
            raise state.load_error
        raise ModelNotAvailableError("No model is loaded.")
    return state.engine


async def read_upload(file: UploadFile, settings: Settings) -> bytes:
    """Read an uploaded image, enforcing the declared type and size limit.

    The declared content type is only a first filter; the engine identifies
    the real format from the file's magic bytes before decoding.
    """
    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE",
                       "Unsupported image type. Use image/jpeg, image/png or image/webp.")
    data = await file.read(settings.max_upload_bytes + 1)
    await file.close()
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "The image exceeds the maximum upload size.")
    if not data:
        raise ApiError(400, "INVALID_IMAGE", "The uploaded image is empty.")
    return data


async def run_inference(request: Request, func: Callable[[], T]) -> T:
    state = get_state(request)
    return await anyio.to_thread.run_sync(func, limiter=state.limiter)


def checked(payload: dict[str, Any], contract: str, settings: Settings) -> dict[str, Any]:
    """Validate a response body against its published JSON Schema."""
    if settings.validate_responses:
        try:
            contracts.validate(payload, contract)
        except Exception as exc:
            logger.error("response failed schema validation: contract=%s error=%s",
                         contract, type(exc).__name__)
            raise ApiError(500, "INTERNAL_ERROR",
                           "The response did not match its schema.") from exc
    return payload
