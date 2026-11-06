"""Página web de chat (RF-05): `GET /` y los archivos estáticos en `/static`."""

from pathlib import Path

from fastapi import APIRouter, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

router = APIRouter(tags=["web"])


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", media_type="text/html; charset=utf-8")


def mount_static(app: FastAPI) -> None:
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
