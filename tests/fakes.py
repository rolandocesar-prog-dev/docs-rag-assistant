"""Dobles de prueba de los puertos (design.md, sección 7): sin red ni base de datos."""

import hashlib
from collections.abc import Sequence

from rag_assistant.domain.document import Document
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.retrieval import EMBEDDING_DIMENSIONS, EmbeddedChunk, RetrievedChunk


class FakeEmbedder:
    """Vector determinista por texto; puede fallar a propósito para probar errores."""

    def __init__(self, *, dimensions: int = EMBEDDING_DIMENSIONS, fail: bool = False) -> None:
        self.dimensions = dimensions
        self.fail = fail
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls.append(list(texts))
        if self.fail:
            raise EmbeddingError("Ollama no responde (simulado).")
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> tuple[float, ...]:
        seed = hashlib.sha256(text.encode()).digest()
        return tuple((seed[i % len(seed)] + 1) / 256 for i in range(self.dimensions))


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.documents: dict[str, tuple[Document, list[EmbeddedChunk]]] = {}

    def content_hash(self, source_path: str) -> str | None:
        stored = self.documents.get(source_path)
        return stored[0].content_hash if stored else None

    def replace_document(self, document: Document, chunks: Sequence[EmbeddedChunk]) -> None:
        self.documents[document.source_path] = (document, list(chunks))

    def search(self, embedding: Sequence[float], *, k: int) -> list[RetrievedChunk]:
        raise NotImplementedError
