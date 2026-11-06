"""Traducción de errores del dominio y de la infraestructura a respuestas HTTP (design.md)."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.exc import OperationalError

from rag_assistant.config import ConfigurationError
from rag_assistant.domain.errors import EmbeddingError, InvalidQuestionError, LLMError

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
        return JSONResponse(status_code=status_code, content={"detail": str(error)})

    return handle


async def _database_unavailable(_request: Request, error: Exception) -> JSONResponse:
    cause = error.orig if isinstance(error, OperationalError) else error
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"La base de datos no responde: {cause}"},
    )
