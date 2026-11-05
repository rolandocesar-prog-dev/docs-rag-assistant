"""Integración: ingesta idempotente sobre PostgreSQL + pgvector real (embeddings simulados)."""

from pathlib import Path

import pytest
from sqlalchemy import Engine, text

from fakes import FakeEmbedder
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ingest import IngestService, IngestStatus
from rag_assistant.cli import ingest as cli
from rag_assistant.config import get_settings


def count(engine: Engine, table: str) -> int:
    with engine.connect() as connection:
        return int(connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())


def test_ingestion_is_idempotent_and_replaces_changed_documents(
    engine: Engine, tmp_path: Path
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "pods.md").write_text("# Pods\n\nUn Pod.\n\n## Red\n\nIP propia.", encoding="utf-8")
    (root / "nodos.md").write_text("# Nodos\n\nMáquinas del clúster.", encoding="utf-8")
    service = IngestService([MarkdownLoader()], FakeEmbedder(), PgVectorStore(engine))

    first = service.ingest(root, license="CC-BY-4.0")
    second = service.ingest(root, license="CC-BY-4.0")
    (root / "pods.md").write_text("# Pods\n\nVersión nueva.", encoding="utf-8")
    third = service.ingest(root, license="CC-BY-4.0")

    assert first.count(IngestStatus.INGESTED) == 2
    assert second.count(IngestStatus.UNCHANGED) == 2
    assert third.count(IngestStatus.INGESTED) == 1
    assert (count(engine, "documents"), count(engine, "chunks")) == (2, 2)


def test_cli_main_wires_the_real_adapters_and_stops_if_ollama_is_down(
    engine: Engine,
    test_database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "pods.md").write_text("# Pods\n\nUn Pod.", encoding="utf-8")
    monkeypatch.setenv("RAG_DATABASE_URL", test_database_url)
    monkeypatch.setenv("RAG_OLLAMA_URL", "http://127.0.0.1:9")
    get_settings.cache_clear()

    code = cli.main([str(tmp_path), "--license", "MIT"])

    get_settings.cache_clear()
    assert code == cli.EXIT_FAILURE
    assert "Error de embeddings" in capsys.readouterr().err
    assert count(engine, "documents") == 0
