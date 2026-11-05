from pathlib import Path

import pytest

from fakes import FakeEmbedder, InMemoryVectorStore
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.loaders.text import TextLoader
from rag_assistant.application.ingest import FileResult, IngestService, IngestStatus
from rag_assistant.domain.errors import EmbeddingError


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    write(root / "pods.md", '---\ntitle: "Pods"\n---\n## Qué es\n\nUn Pod agrupa contenedores.')
    write(root / "red" / "service.md", "# Service\n\nExpone Pods.\n\n## Tipos\n\nClusterIP.")
    write(root / "notas.txt", "Notas en texto plano.")
    write(root / "_index.md", '---\ntitle: "Índice"\n---\n')
    write(root / "informe.docx", "binario")
    write(root / ".oculto" / "borrador.md", "# No debe ingestarse")
    write(root / ".DS_Store", "basura del sistema")
    return root


def service(
    store: InMemoryVectorStore, embedder: FakeEmbedder, results: list[FileResult] | None = None
) -> IngestService:
    return IngestService(
        [MarkdownLoader(), TextLoader()],
        embedder,
        store,
        on_result=results.append if results is not None else None,
    )


def statuses(results: list[FileResult]) -> dict[str, IngestStatus]:
    return {result.source_path: result.status for result in results}


def test_first_ingestion_stores_supported_files_and_reports_the_rest(corpus: Path) -> None:
    store, notified = InMemoryVectorStore(), list[FileResult]()

    report = service(store, FakeEmbedder(), notified).ingest(corpus, license="MIT")

    assert statuses(report.results) == {
        "corpus/_index.md": IngestStatus.EMPTY,
        "corpus/informe.docx": IngestStatus.UNSUPPORTED,
        "corpus/notas.txt": IngestStatus.INGESTED,
        "corpus/pods.md": IngestStatus.INGESTED,
        "corpus/red/service.md": IngestStatus.INGESTED,
    }
    assert notified == report.results
    assert sorted(store.documents) == [
        "corpus/notas.txt",
        "corpus/pods.md",
        "corpus/red/service.md",
    ]
    document, chunks = store.documents["corpus/red/service.md"]
    assert document.license == "MIT"
    assert [c.chunk.section for c in chunks] == ["Service", "Service > Tipos"]
    assert all(c.chunk.document_id == document.id for c in chunks)
    service_result = next(r for r in report.results if r.source_path == "corpus/red/service.md")
    assert service_result.chunks == 2


def test_second_ingestion_skips_unchanged_files_without_calling_the_embedder(
    corpus: Path,
) -> None:
    store = InMemoryVectorStore()
    service(store, FakeEmbedder()).ingest(corpus, license="MIT")
    embedder = FakeEmbedder()

    report = service(store, embedder).ingest(corpus, license="MIT")

    assert report.count(IngestStatus.UNCHANGED) == 3
    assert report.count(IngestStatus.INGESTED) == 0
    assert embedder.calls == []


def test_modified_file_is_ingested_again(corpus: Path) -> None:
    store = InMemoryVectorStore()
    service(store, FakeEmbedder()).ingest(corpus, license="MIT")
    write(corpus / "pods.md", '---\ntitle: "Pods"\n---\nTexto nuevo sobre Pods.')

    report = service(store, FakeEmbedder()).ingest(corpus, license="MIT")

    assert statuses(report.results)["corpus/pods.md"] is IngestStatus.INGESTED
    assert report.count(IngestStatus.UNCHANGED) == 2
    _, chunks = store.documents["corpus/pods.md"]
    assert chunks[0].chunk.content == "Texto nuevo sobre Pods."


def test_a_single_file_uses_its_name_as_source_path(corpus: Path) -> None:
    store = InMemoryVectorStore()

    report = service(store, FakeEmbedder()).ingest(corpus / "red" / "service.md", license="MIT")

    assert [r.source_path for r in report.results] == ["service.md"]


def test_embedding_error_stops_the_ingestion(corpus: Path) -> None:
    store = InMemoryVectorStore()

    with pytest.raises(EmbeddingError):
        service(store, FakeEmbedder(fail=True)).ingest(corpus, license="MIT")
    assert store.documents == {}
