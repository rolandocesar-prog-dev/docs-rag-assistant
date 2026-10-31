"""Punto de entrada de la API. Ejecutar con: uvicorn rag_assistant.main:app --reload"""

from fastapi import FastAPI

from rag_assistant import __version__
from rag_assistant.api.health import router as health_router
from rag_assistant.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(health_router)
    return app


app = create_app()
