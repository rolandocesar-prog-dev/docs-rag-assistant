"""Almacén vectorial sobre PostgreSQL + pgvector, con SQL explícito (design.md, sección 3)."""

from collections.abc import Sequence

from sqlalchemy import Engine, text

from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.document import Document
from rag_assistant.domain.retrieval import EmbeddedChunk, RetrievedChunk

_SELECT_HASH = text("SELECT content_hash FROM documents WHERE source_path = :source_path")
_DELETE_DOCUMENT = text("DELETE FROM documents WHERE source_path = :source_path")
_INSERT_DOCUMENT = text(
    "INSERT INTO documents (id, source_path, title, license, content_hash) "
    "VALUES (:id, :source_path, :title, :license, :content_hash)"
)
_INSERT_CHUNK = text(
    "INSERT INTO chunks (id, document_id, position, section, page, content, embedding) "
    "VALUES (:id, :document_id, :position, :section, :page, :content, CAST(:embedding AS vector))"
)
# `<=>` es la distancia de coseno; ordenar por ella usa el índice HNSW.
_SEARCH = text(
    "SELECT c.id, c.document_id, c.position, c.section, c.page, c.content, "
    "d.title, d.source_path, 1 - (c.embedding <=> CAST(:embedding AS vector)) AS score "
    "FROM chunks c JOIN documents d ON d.id = c.document_id "
    "ORDER BY c.embedding <=> CAST(:embedding AS vector) "
    "LIMIT :k"
)


class PgVectorStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def content_hash(self, source_path: str) -> str | None:
        with self._engine.connect() as connection:
            value = connection.execute(_SELECT_HASH, {"source_path": source_path}).scalar()
        return None if value is None else str(value)

    def replace_document(self, document: Document, chunks: Sequence[EmbeddedChunk]) -> None:
        foreign = [item.chunk.position for item in chunks if item.chunk.document_id != document.id]
        if foreign:
            raise ValueError(f"Fragmentos de otro documento en las posiciones {foreign}.")
        with self._engine.begin() as connection:
            connection.execute(_DELETE_DOCUMENT, {"source_path": document.source_path})
            connection.execute(
                _INSERT_DOCUMENT,
                {
                    "id": document.id,
                    "source_path": document.source_path,
                    "title": document.title,
                    "license": document.license,
                    "content_hash": document.content_hash,
                },
            )
            if chunks:
                connection.execute(
                    _INSERT_CHUNK,
                    [
                        {
                            "id": item.chunk.id,
                            "document_id": item.chunk.document_id,
                            "position": item.chunk.position,
                            "section": item.chunk.section,
                            "page": item.chunk.page,
                            "content": item.chunk.content,
                            "embedding": _to_vector(item.embedding),
                        }
                        for item in chunks
                    ],
                )

    def search(self, embedding: Sequence[float], *, k: int) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k debe ser mayor o igual a 1.")
        with self._engine.connect() as connection:
            rows = connection.execute(_SEARCH, {"embedding": _to_vector(embedding), "k": k})
            return [
                RetrievedChunk(
                    chunk=Chunk(
                        id=row.id,
                        document_id=row.document_id,
                        position=row.position,
                        section=row.section,
                        page=row.page,
                        content=row.content,
                    ),
                    document_title=row.title,
                    source_path=row.source_path,
                    score=float(row.score),
                )
                for row in rows
            ]


def _to_vector(values: Sequence[float]) -> str:
    """Formato de texto de pgvector: `[0.1,0.2,…]`; `repr` conserva toda la precisión."""
    return "[" + ",".join(repr(float(value)) for value in values) + "]"
