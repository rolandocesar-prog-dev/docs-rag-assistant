"""Puerto del proveedor de embeddings: convierte textos en vectores (RF-01, RF-04)."""

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Un vector por texto, en el mismo orden. Lanza `EmbeddingError` si algo falla."""
        ...
