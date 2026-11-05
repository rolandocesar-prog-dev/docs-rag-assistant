"""Lectura y normalización de texto comunes a todos los cargadores."""

import re
from pathlib import Path

_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


def read_text(path: Path) -> str:
    """Lee UTF-8 (con o sin BOM) y unifica los fines de línea en `\\n`."""
    return _unify_newlines(path.read_bytes().decode("utf-8-sig"))


def normalize(text: str) -> str:
    """Quita espacios al final de cada línea y deja como máximo una línea en blanco seguida."""
    lines = "\n".join(line.rstrip() for line in _unify_newlines(text).split("\n"))
    return _EXTRA_BLANK_LINES.sub("\n\n", lines).strip()


def _unify_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")
