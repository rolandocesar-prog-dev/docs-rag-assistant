"""Embeddings con Ollama (`POST /api/embed`), por lotes y con validación de la respuesta."""

from collections.abc import Sequence

import httpx2

from rag_assistant.adapters.ollama_http import new_client, post_json
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.retrieval import EMBEDDING_DIMENSIONS


class OllamaEmbeddings:
    def __init__(
        self,
        client: httpx2.Client,
        *,
        model: str,
        batch_size: int = 32,
        dimensions: int = EMBEDDING_DIMENSIONS,
    ) -> None:
        if batch_size < 1:
            raise ValueError("El tamaño de lote debe ser mayor o igual a 1.")
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._dimensions = dimensions

    @classmethod
    def from_url(cls, base_url: str, *, model: str, batch_size: int = 32) -> "OllamaEmbeddings":
        return cls(new_client(base_url), model=model, batch_size=batch_size)

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[tuple[float, ...]]:
        payload = post_json(
            self._client,
            "/api/embed",
            {"model": self._model, "input": list(batch)},
            model=self._model,
            error=EmbeddingError,
        )
        return self._parse(payload, expected=len(batch))

    def _parse(self, payload: object, *, expected: int) -> list[tuple[float, ...]]:
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != expected:
            raise EmbeddingError(f"Ollama no devolvió {expected} embeddings.")
        vectors: list[tuple[float, ...]] = []
        for vector in embeddings:
            if not isinstance(vector, list) or len(vector) != self._dimensions:
                size = len(vector) if isinstance(vector, list) else "?"
                raise EmbeddingError(
                    f"El modelo «{self._model}» devolvió vectores de {size} dimensiones; "
                    f"el esquema espera {self._dimensions}."
                )
            vectors.append(tuple(float(value) for value in vector))
        return vectors
