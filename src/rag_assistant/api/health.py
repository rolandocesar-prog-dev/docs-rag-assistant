"""Endpoint de salud, usado por Docker, el despliegue y el monitoreo."""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_assistant import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)
