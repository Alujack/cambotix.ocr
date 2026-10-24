"""Enforce the request body limit while streaming, before it is buffered.

The ``Content-Length`` header is checked first; bodies without one (chunked
uploads) are counted as they arrive and rejected as soon as they exceed the
limit.
"""

from __future__ import annotations

import json

from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class PayloadTooLargeError(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413, detail="The request body is too large.")


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope["headers"]).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            await self._reject(scope, send)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise PayloadTooLargeError()
            return message

        await self.app(scope, limited_receive, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        error = {"code": "PAYLOAD_TOO_LARGE", "message": "The request body is too large."}
        request_id = scope.get("state", {}).get("request_id")
        if request_id:
            error["request_id"] = request_id
        body = json.dumps({"error": error}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})
