"""Fragmentos con su vector (para indexar) y resultados de búsqueda (para responder y citar)."""

import math
from dataclasses import dataclass

from rag_assistant.domain.chunk import Chunk


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.embedding:
            raise ValueError("El embedding no puede estar vacío.")
        if not all(math.isfinite(value) for value in self.embedding):
            raise ValueError("El embedding solo admite números finitos.")


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Resultado de una búsqueda; `score` es la similitud de coseno (1 = idéntico)."""

    chunk: Chunk
    document_title: str
    source_path: str
    score: float
