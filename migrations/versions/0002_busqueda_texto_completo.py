"""Texto completo en español para la búsqueda híbrida (design.md, almacén vectorial).

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-05
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | Sequence[str] | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Columna generada: PostgreSQL la calcula para las filas existentes y las nuevas.
    op.execute(
        """
        ALTER TABLE chunks ADD COLUMN content_tsv tsvector
            GENERATED ALWAYS AS (
                to_tsvector('spanish', coalesce(section, '') || ' ' || content)
            ) STORED
        """
    )
    op.execute("CREATE INDEX chunks_content_tsv_gin ON chunks USING gin (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX chunks_content_tsv_gin")
    op.execute("ALTER TABLE chunks DROP COLUMN content_tsv")
