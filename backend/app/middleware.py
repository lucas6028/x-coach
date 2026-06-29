"""ASGI middleware: reject over-large request bodies before they are buffered.

FastAPI parses (and spools) the entire multipart body while resolving the ``UploadFile``
dependency — i.e. *before* the route handler runs. So the only place to reject an oversized
upload before it is written to the spool (consuming RAM/disk) is here, ahead of routing.

This uses the declared ``Content-Length`` as a cheap pre-check and rejects with HTTP 413 without
reading the body. It is a fast, well-understood guard (the same thing ``client_max_body_size``
does at a reverse proxy); the upload handler additionally enforces the limit while streaming, as
a backstop for requests that lie about or omit ``Content-Length``.
"""

from __future__ import annotations

from typing import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

_TOO_LARGE_BODY = b'{"detail":"Uploaded file is too large."}'


class BodySizeLimitMiddleware:
    """Reject HTTP requests whose declared body size exceeds ``max_body_bytes``.

    ``max_body_bytes`` may be an int or a zero-arg callable returning an int (so the limit can
    be read from config dynamically — and overridden in tests — rather than frozen at startup).
    """

    def __init__(self, app: ASGIApp, *, max_body_bytes: int | Callable[[], int]) -> None:
        self.app = app
        self._max_body_bytes = max_body_bytes

    @property
    def max_body_bytes(self) -> int:
        limit = self._max_body_bytes
        return limit() if callable(limit) else limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            declared = _content_length(scope)
            if declared is not None and declared > self.max_body_bytes:
                await self._reject(send)
                return
        await self.app(scope, receive, send)

    async def _reject(self, send: Send) -> None:
        start: Message = {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_TOO_LARGE_BODY)).encode("ascii")),
            ],
        }
        await send(start)
        await send({"type": "http.response.body", "body": _TOO_LARGE_BODY})


def _content_length(scope: Scope) -> int | None:
    """Return the request's ``Content-Length`` as an int, or ``None`` if absent/unparseable."""
    for name, value in scope.get("headers") or []:
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None
