"""Dobles de prueba de los puertos (design.md, sección 7): sin red ni base de datos."""

import hashlib
from collections.abc import Sequence

from rag_assistant.domain.completion import Completion
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
    """Guarda documentos en memoria; `search` devuelve los resultados que fije la prueba."""

    def __init__(self, search_results: Sequence[RetrievedChunk] = ()) -> None:
        self.documents: dict[str, tuple[Document, list[EmbeddedChunk]]] = {}
        self.search_results = list(search_results)
        self.search_ks: list[int] = []

    def content_hash(self, source_path: str) -> str | None:
        stored = self.documents.get(source_path)
        return stored[0].content_hash if stored else None

    def replace_document(self, document: Document, chunks: Sequence[EmbeddedChunk]) -> None:
        self.documents[document.source_path] = (document, list(chunks))

    def search(self, embedding: Sequence[float], *, k: int) -> list[RetrievedChunk]:
        self.search_ks.append(k)
        return self.search_results[:k]


class FakeLLM:
    """Devuelve una respuesta fija y guarda los mensajes que recibió."""

    def __init__(self, completion: Completion) -> None:
        self.completion = completion
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        return self.completion


class ScriptedLLM:
    """Devuelve las respuestas en orden; una excepción de la lista se lanza en su turno."""

    def __init__(self, *replies: Completion | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> Completion:
        self.calls.append((system, user))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
