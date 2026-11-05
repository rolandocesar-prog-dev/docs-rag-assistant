"""Respuesta al usuario con sus fuentes (RF-02, RF-03)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Citation:
    """Fuente citada; `id` coincide con la marca `[id]` del texto de la respuesta."""

    id: int
    document: str
    section: str | None
    page: int | None
    source_path: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    found: bool
