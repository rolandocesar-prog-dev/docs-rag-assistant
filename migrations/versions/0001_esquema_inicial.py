"""Esquema inicial: documentos y fragmentos con embeddings de bge-m3 (design.md, sección 3).

Revision ID: 0001
Revises:
Create Date: 2026-11-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0001"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE documents (
            id           UUID PRIMARY KEY,
            source_path  TEXT NOT NULL UNIQUE,
            title        TEXT NOT NULL,
            license      TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE chunks (
            id          UUID PRIMARY KEY,
            document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
            position    INT  NOT NULL CHECK (position >= 0),
            section     TEXT,
            page        INT  CHECK (page >= 1),
            content     TEXT NOT NULL,
            embedding   vector(1024) NOT NULL,
            UNIQUE (document_id, position)
        )
        """
    )
    op.execute(
        "CREATE INDEX chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    op.execute("DROP EXTENSION IF EXISTS vector")
