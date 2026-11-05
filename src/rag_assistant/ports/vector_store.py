"""Puerto del almacén vectorial: guarda documentos con sus fragmentos y busca por similitud."""

from collections.abc import Sequence
from typing import Protocol

from rag_assistant.domain.document import Document
from rag_assistant.domain.retrieval import EmbeddedChunk, RetrievedChunk


class VectorStore(Protocol):
    def content_hash(self, source_path: str) -> str | None:
        """Hash guardado del documento, o `None` si nunca se ingestó."""
        ...

    def replace_document(self, document: Document, chunks: Sequence[EmbeddedChunk]) -> None:
        """Reemplaza la versión anterior del documento y sus fragmentos en una transacción."""
        ...

    def search(self, embedding: Sequence[float], *, k: int) -> list[RetrievedChunk]:
        """Los `k` fragmentos más similares, de mayor a menor `score`."""
        ...
