"""De punta a punta: ingesta y pregunta con PostgreSQL + pgvector y Ollama reales.

Requiere RAG_TEST_DATABASE_URL y RAG_TEST_OLLAMA_URL (con bge-m3 y qwen3.5:4b); si falta alguna,
se omite. En CI no corre, porque no hay Ollama.
"""

from pathlib import Path

import pytest
from sqlalchemy import Engine

from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.adapters.llm.ollama import OllamaChat
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ask import AskService
from rag_assistant.application.ingest import IngestService
from rag_assistant.domain.prompting import NOT_FOUND


@pytest.fixture
def ask_service(engine: Engine, test_ollama_url: str, tmp_path: Path) -> AskService:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "pods.md").write_text(
        '---\ntitle: "Pods"\n---\n## ¿Qué es un Pod?\n\nUn Pod es la unidad de despliegue más '
        "pequeña de Kubernetes: agrupa uno o más contenedores que comparten red y almacenamiento.",
        encoding="utf-8",
    )
    embedder = OllamaEmbeddings.from_url(test_ollama_url, model="bge-m3")
    store = PgVectorStore(engine)
    IngestService([MarkdownLoader()], embedder, store).ingest(docs, license="CC-BY-4.0")
    return AskService(embedder, store, OllamaChat.from_url(test_ollama_url, model="qwen3.5:4b"))


def test_answers_a_question_about_the_documents_with_a_citation(ask_service: AskService) -> None:
    answer = ask_service.ask("¿Qué es un Pod?")

    assert answer.found
    assert "[1]" in answer.text
    [citation] = answer.citations
    assert (citation.document, citation.source_path) == ("Pods", "docs/pods.md")


def test_off_topic_question_is_not_answered(ask_service: AskService) -> None:
    answer = ask_service.ask("¿Cuál es la capital de Francia?")

    assert (answer.found, answer.text) == (False, NOT_FOUND)
