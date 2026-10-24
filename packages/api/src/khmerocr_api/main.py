"""FastAPI application.

Run with ``khmerocr serve`` or ``uvicorn khmerocr_api.main:app``. The service
binds to 127.0.0.1 by default; it has no authentication and must not be
exposed to untrusted networks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from khmerocr import KhmerOCR
from starlette.formparsers import MultiPartParser

from ._version import __version__
from .dependencies import EngineState, Settings
from .errors import register_error_handlers
from .middleware import BodySizeLimitMiddleware, RequestIDMiddleware
from .routes import documents, health, models, ocr

API_PREFIX = "/v1"
# Allowance for multipart boundaries and form fields around the image.
_MULTIPART_OVERHEAD = 64 * 1024


def create_app(*, settings: Settings | None = None, engine: KhmerOCR | None = None) -> FastAPI:
    """Build the app. Pass ``engine`` to serve a preloaded engine (tests,
    embedding); otherwise the model named in ``settings`` is loaded at startup."""
    settings = settings or Settings.from_env()

    # Keep uploads in memory: by default Starlette spools file parts larger
    # than 1 MiB to temporary files on disk, which would write ID images to
    # disk. This is process-wide, so run the API in its own process.
    MultiPartParser.spool_max_size = settings.max_upload_bytes + _MULTIPART_OVERHEAD

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = EngineState(settings=settings, engine=engine)
        if engine is None:
            state.load()
        app.state.engine_state = state
        try:
            yield
        finally:
            state.close()

    app = FastAPI(
        title="KhmerOCR API",
        version=__version__,
        description="Local Khmer/English OCR and Cambodian National ID extraction.",
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    register_error_handlers(app)
    for router in (health.router, models.router, ocr.router, documents.router):
        app.include_router(router, prefix=API_PREFIX)

    # Added last = outermost: request IDs are assigned before size checks run.
    app.add_middleware(BodySizeLimitMiddleware,
                       max_bytes=settings.max_upload_bytes + _MULTIPART_OVERHEAD)
    app.add_middleware(RequestIDMiddleware)
    return app


app = create_app()
