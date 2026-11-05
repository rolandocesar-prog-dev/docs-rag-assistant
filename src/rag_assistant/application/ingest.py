"""Caso de uso de ingesta (RF-01): cargar, dividir, generar embeddings y guardar lo que cambió."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

from rag_assistant.domain.chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, split_document
from rag_assistant.domain.errors import EmptyDocumentError
from rag_assistant.domain.retrieval import EmbeddedChunk
from rag_assistant.ports.document_loader import DocumentLoader
from rag_assistant.ports.embedding_provider import EmbeddingProvider
from rag_assistant.ports.vector_store import VectorStore


class IngestStatus(StrEnum):
    INGESTED = "ingestado"
    UNCHANGED = "sin cambios"
    EMPTY = "vacío"
    UNSUPPORTED = "formato no soportado"


@dataclass(frozen=True, slots=True)
class FileResult:
    source_path: str
    status: IngestStatus
    chunks: int = 0


@dataclass(slots=True)
class IngestReport:
    results: list[FileResult] = field(default_factory=list)

    def count(self, status: IngestStatus) -> int:
        return sum(result.status is status for result in self.results)


class IngestService:
    def __init__(
        self,
        loaders: Sequence[DocumentLoader],
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap: int = DEFAULT_OVERLAP,
        on_result: Callable[[FileResult], None] | None = None,
    ) -> None:
        self._loaders = loaders
        self._embedder = embedder
        self._store = store
        self._max_chars = max_chars
        self._overlap = overlap
        self._on_result = on_result

    def ingest(self, root: Path, *, license: str) -> IngestReport:
        """Ingesta un archivo o una carpeta. Un `EmbeddingError` detiene el proceso."""
        report = IngestReport()
        for path in _files(root):
            result = self._ingest_file(path, root.parent, license)
            report.results.append(result)
            if self._on_result is not None:
                self._on_result(result)
        return report

    def _ingest_file(self, path: Path, base: Path, license: str) -> FileResult:
        source_path = path.relative_to(base).as_posix()
        loader = next((loader for loader in self._loaders if loader.supports(path)), None)
        if loader is None:
            return FileResult(source_path, IngestStatus.UNSUPPORTED)
        try:
            document = replace(loader.load(path, license=license), source_path=source_path)
        except EmptyDocumentError:
            return FileResult(source_path, IngestStatus.EMPTY)
        if self._store.content_hash(source_path) == document.content_hash:
            return FileResult(source_path, IngestStatus.UNCHANGED)

        chunks = split_document(document, max_chars=self._max_chars, overlap=self._overlap)
        vectors = self._embedder.embed([chunk.content for chunk in chunks])
        embedded = [EmbeddedChunk(c, v) for c, v in zip(chunks, vectors, strict=True)]
        self._store.replace_document(document, embedded)
        return FileResult(source_path, IngestStatus.INGESTED, chunks=len(chunks))


def _files(root: Path) -> list[Path]:
    """El archivo, o los archivos de la carpeta en orden, sin los ocultos (que empiezan con `.`)."""
    if root.is_file():
        return [root]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
