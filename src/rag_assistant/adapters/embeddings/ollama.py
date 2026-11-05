"""Embeddings con Ollama (`POST /api/embed`), por lotes y con validación de la respuesta."""

from collections.abc import Sequence

import httpx2

from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.retrieval import EMBEDDING_DIMENSIONS

# La primera llamada carga el modelo en memoria; en CPU puede tardar.
DEFAULT_TIMEOUT_SECONDS = 300.0


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
        client = httpx2.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT_SECONDS)
        return cls(client, model=model, batch_size=batch_size)

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]))
        return vectors

    def _embed_batch(self, batch: Sequence[str]) -> list[tuple[float, ...]]:
        try:
            response = self._client.post(
                "/api/embed", json={"model": self._model, "input": list(batch)}
            )
        except httpx2.HTTPError as error:
            raise EmbeddingError(
                f"Ollama no responde en {self._client.base_url}: {error}"
            ) from error
        if response.status_code == httpx2.codes.NOT_FOUND:
            raise EmbeddingError(
                f"Ollama no tiene el modelo «{self._model}». Descárgalo con: "
                f"docker compose exec ollama ollama pull {self._model}"
            )
        if response.is_error:
            raise EmbeddingError(f"Ollama respondió {response.status_code}: {response.text[:200]}")
        return self._parse(response.json(), expected=len(batch))

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
