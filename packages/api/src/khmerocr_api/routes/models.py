from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from khmerocr.models import list_installed_models

from ..dependencies import get_state

router = APIRouter(tags=["service"])


@router.get("/models")
def models(request: Request) -> dict[str, Any]:
    """List installed model bundles and the model serving requests."""
    state = get_state(request)
    active = state.engine.info() if state.engine is not None else None
    return {
        "active": active,
        "installed": [m.summary() for m in list_installed_models()],
    }
