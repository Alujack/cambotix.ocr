"""Require an API key on every endpoint except the health check.

Clients send ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. The
check runs before the request body is read, so an unauthenticated client
cannot upload an image at all. ``/v1/health`` stays open for container and
load-balancer health checks; it reports no document data.
"""

from __future__ import annotations

import json
import secrets

from starlette.types import ASGIApp, Receive, Scope, Send

OPEN_PATHS = frozenset({"/v1/health"})


class ApiKeyMiddleware:
    def __init__(self, app: ASGIApp, *, api_key: str) -> None:
        self.app = app
        self._key = api_key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        presented = headers.get(b"x-api-key", b"")
        authorization = headers.get(b"authorization", b"")
        if not presented and authorization[:7].lower() == b"bearer ":
            presented = authorization[7:].strip()
        if presented and secrets.compare_digest(presented, self._key):
            await self.app(scope, receive, send)
            return

        error: dict[str, str] = {"code": "UNAUTHORIZED",
                                 "message": "A valid API key is required."}
        request_id = scope.get("state", {}).get("request_id")
        if request_id:
            error["request_id"] = request_id
        body = json.dumps({"error": error}).encode()
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"www-authenticate", b"Bearer"),
        ]})
        await send({"type": "http.response.body", "body": body})
