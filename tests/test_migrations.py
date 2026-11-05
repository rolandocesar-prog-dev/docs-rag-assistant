"""Integración: la migración inicial crea y elimina el esquema de design.md en PostgreSQL real."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, create_engine, inspect, text
from sqlalchemy.exc import DataError, IntegrityError


@pytest.fixture
def migrated(test_database_url: str, alembic_config: Config) -> Iterator[Connection]:
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")
    engine = create_engine(test_database_url)
    with engine.connect() as connection:
        yield connection
    engine.dispose()


def insert_document(connection: Connection) -> str:
    document_id = str(uuid4())
    connection.execute(
        text(
            "INSERT INTO documents (id, source_path, title, license, content_hash) "
            "VALUES (:id, :path, 'Pods', 'CC-BY-4.0', 'hash')"
        ),
        {"id": document_id, "path": f"{document_id}.md"},
    )
    return document_id


def insert_chunk(connection: Connection, document_id: str, position: int, dims: int) -> None:
    embedding = "[" + ",".join(["0.1"] * dims) + "]"
    connection.execute(
        text(
            "INSERT INTO chunks (id, document_id, position, content, embedding) "
            "VALUES (:id, :document_id, :position, 'texto', :embedding)"
        ),
        {
            "id": str(uuid4()),
            "document_id": document_id,
            "position": position,
            "embedding": embedding,
        },
    )


def test_upgrade_creates_tables_index_and_vector_column(migrated: Connection) -> None:
    inspector = inspect(migrated)

    assert {"documents", "chunks"} <= set(inspector.get_table_names())
    embedding_type = migrated.execute(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'"
        )
    ).scalar_one()
    assert embedding_type == "vector(1024)"
    index = migrated.execute(
        text("SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'")
    ).scalar_one()
    assert "hnsw" in index and "vector_cosine_ops" in index


def test_schema_enforces_dimension_uniqueness_and_cascade(migrated: Connection) -> None:
    with migrated.begin():
        document_id = insert_document(migrated)
        insert_chunk(migrated, document_id, position=0, dims=1024)

    with pytest.raises(DataError), migrated.begin():
        insert_chunk(migrated, document_id, position=1, dims=3)
    with pytest.raises(IntegrityError), migrated.begin():
        insert_chunk(migrated, document_id, position=0, dims=1024)

    with migrated.begin():
        migrated.execute(text("DELETE FROM documents WHERE id = :id"), {"id": document_id})
    remaining = migrated.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    assert remaining == 0


def test_downgrade_removes_the_schema(test_database_url: str, alembic_config: Config) -> None:
    command.upgrade(alembic_config, "head")

    command.downgrade(alembic_config, "base")

    engine = create_engine(test_database_url)
    with engine.connect() as connection:
        assert not {"documents", "chunks"} & set(inspect(connection).get_table_names())
    engine.dispose()
    command.upgrade(alembic_config, "head")
