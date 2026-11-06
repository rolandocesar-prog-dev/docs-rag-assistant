"""De punta a punta: `rag-eval` con PostgreSQL + pgvector y Ollama reales.

Requiere RAG_TEST_DATABASE_URL y RAG_TEST_OLLAMA_URL (con bge-m3 y qwen3.5:4b); si falta alguna,
se omite. En CI no corre, porque no hay Ollama.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine

from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ingest import IngestService
from rag_assistant.cli import evaluate as cli
from rag_assistant.config import get_settings


@pytest.fixture
def evaluation_env(
    engine: Engine,
    test_database_url: str,
    test_ollama_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pods.md").write_text(
        '---\ntitle: "Pods"\n---\n## ¿Qué es un Pod?\n\nUn Pod es la unidad de despliegue más '
        "pequeña de Kubernetes: agrupa uno o más contenedores que comparten red y almacenamiento.",
        encoding="utf-8",
    )
    embedder = OllamaEmbeddings.from_url(test_ollama_url, model="bge-m3")
    IngestService([MarkdownLoader()], embedder, PgVectorStore(engine)).ingest(docs, license="MIT")
    items = [
        {"id": "q01", "category": "definicion", "question": "¿Qué es un Pod?",
         "expected": [{"source_path": "docs/pods.md", "section": "¿Qué es un Pod?"}],
         "reference": "Un Pod agrupa uno o más contenedores que comparten red y almacenamiento."},
        {"id": "q02", "category": "fuera_de_tema", "question": "¿Cuál es la capital de Francia?",
         "expected": [], "reference": None},
    ]  # fmt: skip
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8"
    )
    for name, value in {
        "RAG_DATABASE_URL": test_database_url,
        "RAG_OLLAMA_URL": test_ollama_url,
        "RAG_LLM_PROVIDER": "ollama",
        "RAG_LLM_MODEL": "",
    }.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()
    yield dataset
    get_settings.cache_clear()


def test_rag_eval_measures_the_real_system_and_writes_the_report(
    evaluation_env: Path, tmp_path: Path
) -> None:
    code = cli.main(["--dataset", str(evaluation_env), "--salida", str(tmp_path / "reports")])

    assert code == cli.EXIT_OK
    [report_json] = (tmp_path / "reports").glob("*.json")
    data = json.loads(report_json.read_text(encoding="utf-8"))
    assert data["llm"] == "ollama/qwen3.5:4b"
    first, second = data["results"]
    assert first["doc_rank"] == 1 and first["found"] and first["cites_expected"]
    assert not second["found"]
    assert report_json.with_suffix(".md").is_file()
