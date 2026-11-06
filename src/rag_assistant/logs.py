"""Logs estructurados en JSON con el id de la petición en curso (RNF-04, design.md)."""

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal, TextIO

LogFormat = Literal["json", "text"]

# Id de la petición en curso; lo fija el middleware de la API.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

# Atributos propios de `LogRecord`: lo demás llegó por `extra=` y se agrega como campo.
# `color_message` es el mensaje con códigos de color que agrega uvicorn: en JSON es ruido.
_RECORD_ATTRIBUTES = frozenset(
    {
        *vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)),
        "message",
        "asctime",
        "color_message",
    }
)
_TEXT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(request_id)s] %(message)s"
_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error")


class RequestIdFilter(logging.Filter):
    """Agrega a cada registro el `request_id` de la petición en curso (`None` fuera de una)."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            entry["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _RECORD_ATTRIBUTES and key != "request_id":
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if getattr(record, "request_id", None) is None:
            record.request_id = "-"
        return super().format(record)


class _AppHandler(logging.StreamHandler[TextIO]):
    """Marca el handler propio para reemplazarlo sin tocar los demás."""


def configure_logging(log_format: LogFormat = "json", level: str = "INFO") -> None:
    """Un solo handler de la app en el logger raíz; uvicorn usa el mismo formato."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, _AppHandler)]:
        root.removeHandler(handler)
    handler = _AppHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter() if log_format == "json" else _TextFormatter(_TEXT_FORMAT))
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    # La app registra una línea por petición (api/middleware.py).
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
