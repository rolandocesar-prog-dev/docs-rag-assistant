"""Punto de entrada de la API. Ejecutar con: uvicorn rag_assistant.main:app --reload"""

from fastapi import FastAPI

from rag_assistant import __version__
from rag_assistant.api.ask import router as ask_router
from rag_assistant.api.errors import register_error_handlers
from rag_assistant.api.health import router as health_router
from rag_assistant.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(health_router)
    app.include_router(ask_router)
    register_error_handlers(app)
    return app


app = create_app()
