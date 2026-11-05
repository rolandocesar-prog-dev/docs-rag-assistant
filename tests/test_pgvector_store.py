"""Integración: `PgVectorStore` contra PostgreSQL + pgvector real (design.md, almacén vectorial)."""

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import DataError

from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.document import Document, DocumentPart
from rag_assistant.domain.retrieval import EmbeddedChunk
from rag_assistant.ports.vector_store import VectorStore

DIMENSIONS = 1024


def axis(index: int, dims: int = DIMENSIONS) -> tuple[float, ...]:
    """Vector unitario en el eje `index`: dos ejes distintos tienen similitud 0."""
    return tuple(1.0 if position == index else 0.0 for position in range(dims))


def make_document(text_: str = "Un Pod agrupa contenedores.", path: str = "pods.md") -> Document:
    return Document(
        source_path=path, title="Pods", license="CC-BY-4.0", parts=(DocumentPart(text_),)
    )


def embedded(document: Document, *axes: int, page: int | None = None) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            Chunk(document.id, position, f"Fragmento {position}", section="Pods", page=page),
            axis(index),
        )
        for position, index in enumerate(axes)
    ]


@pytest.fixture
def store(engine: Engine) -> VectorStore:
    return PgVectorStore(engine)


def count_chunks(engine: Engine) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text("SELECT count(*) FROM chunks")).scalar_one())


def test_content_hash_is_none_until_the_document_is_stored(store: VectorStore) -> None:
    document = make_document()
    assert store.content_hash(document.source_path) is None

    store.replace_document(document, embedded(document, 0))

    assert store.content_hash(document.source_path) == document.content_hash


def test_search_returns_the_most_similar_chunks_with_citation_data(store: VectorStore) -> None:
    document = make_document()
    store.replace_document(document, embedded(document, 0, 1, 2, page=3))

    # Consulta cercana al eje 1, algo cercana al eje 0 y ortogonal al eje 2.
    query = tuple(a + 0.5 * b for a, b in zip(axis(1), axis(0), strict=True))

    results = store.search(query, k=2)

    assert [result.chunk.position for result in results] == [1, 0]
    best, second = results
    assert best.score == pytest.approx(1 / 1.25**0.5)
    assert second.score == pytest.approx(0.5 / 1.25**0.5)
    assert (best.document_title, best.source_path) == ("Pods", "pods.md")
    assert (best.chunk.section, best.chunk.page, best.chunk.content) == ("Pods", 3, "Fragmento 1")
    assert best.chunk.document_id == document.id


def test_replacing_a_document_removes_its_previous_chunks(
    store: VectorStore, engine: Engine
) -> None:
    first = make_document("Versión 1.")
    store.replace_document(first, embedded(first, 0, 1, 2))
    second = make_document("Versión 2.")

    store.replace_document(second, embedded(second, 3))

    assert count_chunks(engine) == 1
    assert store.content_hash("pods.md") == second.content_hash
    [result] = store.search(axis(3), k=5)
    assert result.chunk.document_id == second.id


def test_failed_replacement_keeps_the_previous_version(store: VectorStore, engine: Engine) -> None:
    first = make_document("Versión 1.")
    store.replace_document(first, embedded(first, 0, 1))
    second = make_document("Versión 2.")
    wrong_size = [EmbeddedChunk(Chunk(second.id, 0, "texto"), axis(0, dims=3))]

    with pytest.raises(DataError):
        store.replace_document(second, wrong_size)

    assert store.content_hash("pods.md") == first.content_hash
    assert count_chunks(engine) == 2


def test_rejects_chunks_from_another_document(store: VectorStore) -> None:
    document, other = make_document(), make_document(path="otro.md")

    with pytest.raises(ValueError, match="otro documento"):
        store.replace_document(document, embedded(other, 0))


def test_documents_are_isolated_by_source_path(store: VectorStore) -> None:
    pods, nodes = make_document(path="pods.md"), make_document(path="nodos.md")
    store.replace_document(pods, embedded(pods, 0))
    store.replace_document(nodes, embedded(nodes, 1))

    results = store.search(axis(1), k=10)

    assert [result.source_path for result in results] == ["nodos.md", "pods.md"]


def test_search_on_an_empty_store_and_invalid_k(store: VectorStore) -> None:
    assert store.search(axis(0), k=5) == []
    with pytest.raises(ValueError, match="k"):
        store.search(axis(0), k=0)
