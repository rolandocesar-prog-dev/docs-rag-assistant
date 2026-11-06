"""Traducción de errores del dominio y de la infraestructura a respuestas HTTP (design.md)."""

import logging
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from rag_assistant.config import ConfigurationError
from rag_assistant.domain.errors import EmbeddingError, InvalidQuestionError, LLMError

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[Exception], int] = {
    InvalidQuestionError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    EmbeddingError: status.HTTP_502_BAD_GATEWAY,
    LLMError: status.HTTP_502_BAD_GATEWAY,
    ConfigurationError: status.HTTP_503_SERVICE_UNAVAILABLE,
}


def register_error_handlers(app: FastAPI) -> None:
    for error_type, status_code in _STATUS_BY_ERROR.items():
        app.add_exception_handler(error_type, _handler(status_code))
    app.add_exception_handler(OperationalError, _database_unavailable)


def _handler(status_code: int) -> Callable[[Request, Exception], Awaitable[JSONResponse]]:
    async def handle(_request: Request, error: Exception) -> JSONResponse:
        if status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            _log_unavailable(error, status_code)
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handle


async def _database_unavailable(_request: Request, error: Exception) -> JSONResponse:
    cause = error.orig if isinstance(error, OperationalError) else error
    _log_unavailable(error, status.HTTP_503_SERVICE_UNAVAILABLE)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"La base de datos no responde: {cause}"},
    )


def _log_unavailable(error: Exception, status_code: int) -> None:
    logger.warning(
        "Servicio externo no disponible",
        extra={"error": type(error).__name__, "detail": str(error), "status": status_code},
    )
