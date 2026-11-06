"""Middleware ASGI: id de petición y una línea de log por petición (RNF-04, design.md)."""

import logging
import re
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from rag_assistant.logs import request_id_var

REQUEST_ID_HEADER = "X-Request-ID"
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
# El healthcheck de Docker llama cada pocos segundos: se registra solo en nivel DEBUG.
_QUIET_PATHS = frozenset({"/health"})

logger = logging.getLogger("rag_assistant.http")


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope) or uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status = 500  # si la app falla antes de responder

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append(REQUEST_ID_HEADER, request_id)
            await send(message)

        path: str = scope["path"]
        try:
            await self._app(scope, receive, send_with_request_id)
        except Exception:
            logger.exception("Error inesperado", extra={"method": scope["method"], "path": path})
            raise
        finally:
            logger.log(
                logging.DEBUG if path in _QUIET_PATHS else logging.INFO,
                "Petición",
                extra={
                    "method": scope["method"],
                    "path": path,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                },
            )
            request_id_var.reset(token)


def _incoming_request_id(scope: Scope) -> str | None:
    """El id que manda el cliente o un proxy, solo si es seguro de registrar y reenviar."""
    value = Headers(scope=scope).get(REQUEST_ID_HEADER)
    return value if value is not None and _VALID_REQUEST_ID.fullmatch(value) else None
