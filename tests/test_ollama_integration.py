"""Integración con Ollama real y bge-m3 (solo si RAG_TEST_OLLAMA_URL está definida)."""

import math

from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.domain.retrieval import EMBEDDING_DIMENSIONS


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def test_bge_m3_returns_vectors_that_capture_meaning_in_spanish(test_ollama_url: str) -> None:
    embeddings = OllamaEmbeddings.from_url(test_ollama_url, model="bge-m3")

    question, related, unrelated = embeddings.embed(
        [
            "¿Qué es un Pod en Kubernetes?",
            "Un Pod es la unidad de despliegue más pequeña: agrupa uno o más contenedores.",
            "La receta lleva harina, huevos y azúcar, y se hornea durante media hora.",
        ]
    )

    assert len(question) == EMBEDDING_DIMENSIONS
    assert cosine(question, related) > cosine(question, unrelated) + 0.1
