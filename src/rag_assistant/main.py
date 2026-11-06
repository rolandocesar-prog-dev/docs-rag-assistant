"""Punto de entrada de la API. Ejecutar con: uvicorn rag_assistant.main:app --reload"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_assistant import __version__
from rag_assistant.adapters.llm.factory import validate_llm_settings
from rag_assistant.api.ask import router as ask_router
from rag_assistant.api.errors import register_error_handlers
from rag_assistant.api.health import router as health_router
from rag_assistant.config import ConfigurationError, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Si falta la configuración del LLM, la app no arranca (RF-04)."""
    try:
        validate_llm_settings(get_settings())
    except ConfigurationError as error:
        logger.error("Configuración inválida: %s", error)
        raise
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(ask_router)
    register_error_handlers(app)
    return app


app = create_app()
