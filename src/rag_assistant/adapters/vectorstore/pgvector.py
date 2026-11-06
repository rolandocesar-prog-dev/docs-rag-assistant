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
# Búsqueda híbrida (design.md): los mejores candidatos por vector y por texto completo en
# español, unidos con Reciprocal Rank Fusion. La parte de texto solo usa los términos raros de
# la pregunta (unidos con O): las palabras frecuentes («cómo», «clúster») traen ruido. La
# frecuencia de cada término se cuenta con el índice GIN.
_HYBRID_SEARCH = text(
    """
    WITH lexemes AS (
        SELECT DISTINCT unnest(tsvector_to_array(to_tsvector('spanish', :text))) AS word
    ),
    rare AS (
        SELECT string_agg(quote_literal(l.word), ' | ') AS terms
        FROM lexemes l
        WHERE (SELECT count(*) FROM chunks c
               WHERE c.content_tsv @@ CAST(quote_literal(l.word) AS tsquery))
              <= greatest(1, :rare_fraction * (SELECT count(*) FROM chunks))
    ),
    query AS (
        SELECT CAST(:embedding AS vector) AS embedding, CAST(r.terms AS tsquery) AS terms
        FROM rare r
    ),
    by_vector AS (
        SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> q.embedding) AS rank
        FROM chunks c, query q
        ORDER BY c.embedding <=> q.embedding
        LIMIT :candidates
    ),
    by_text AS (
        SELECT c.id, row_number() OVER (ORDER BY ts_rank_cd(c.content_tsv, q.terms) DESC) AS rank
        FROM chunks c, query q
        WHERE c.content_tsv @@ q.terms
        ORDER BY ts_rank_cd(c.content_tsv, q.terms) DESC
        LIMIT :text_candidates
    ),
    fused AS (
        SELECT id, sum(1.0 / (:rrf_k + rank)) AS rrf
        FROM (SELECT * FROM by_vector UNION ALL SELECT * FROM by_text) AS ranked
        GROUP BY id
    )
    SELECT c.id, c.document_id, c.position, c.section, c.page, c.content,
           d.title, d.source_path, 1 - (c.embedding <=> q.embedding) AS score
    FROM fused f
    JOIN chunks c ON c.id = f.id
    JOIN documents d ON d.id = c.document_id
    CROSS JOIN query q
    ORDER BY f.rrf DESC, score DESC
    LIMIT :k
    """
)
# Candidatos de cada lista, constante de Reciprocal Rank Fusion (60 es el valor habitual) y
# fracción máxima de fragmentos en que puede aparecer un término para usarlo en la parte de
# texto. Valores medidos con `rag-eval` (design.md, sección 8).
HYBRID_CANDIDATES = 20
TEXT_CANDIDATES = 5
RRF_K = 60
RARE_TERM_FRACTION = 0.02


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

    def search(
        self, embedding: Sequence[float], *, k: int, text: str | None = None
    ) -> list[RetrievedChunk]:
        if k < 1:
            raise ValueError("k debe ser mayor o igual a 1.")
        params: dict[str, object] = {"embedding": _to_vector(embedding), "k": k}
        query = _SEARCH
        if text is not None:
            query = _HYBRID_SEARCH
            params |= {
                "text": text,
                "candidates": max(k, HYBRID_CANDIDATES),
                "text_candidates": TEXT_CANDIDATES,
                "rrf_k": RRF_K,
                "rare_fraction": RARE_TERM_FRACTION,
            }
        with self._engine.connect() as connection:
            rows = connection.execute(query, params)
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
