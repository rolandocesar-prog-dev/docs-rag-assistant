from uuid import uuid4

import pytest

from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.retrieval import EmbeddedChunk


@pytest.mark.parametrize(
    ("embedding", "message"),
    [((), "vacío"), ((0.1, float("nan")), "finitos"), ((float("inf"),), "finitos")],
    ids=["vacio", "nan", "infinito"],
)
def test_embedded_chunk_rejects_invalid_vectors(embedding: tuple[float, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddedChunk(Chunk(uuid4(), 0, "texto"), embedding)
