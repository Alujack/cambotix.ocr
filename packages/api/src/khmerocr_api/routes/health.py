from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from khmerocr import __version__ as sdk_version

from .._version import __version__
from ..dependencies import get_state

router = APIRouter(tags=["service"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Report that the service is running and whether a model is loaded."""
    state = get_state(request)
    body: dict[str, Any] = {
        "status": "ok",
        "model_loaded": state.engine is not None,
        "api_version": __version__,
        "sdk_version": sdk_version,
    }
    if state.load_error is not None:
        body["model_error"] = state.load_error.code.value
    return body
