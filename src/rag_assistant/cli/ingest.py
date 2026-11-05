"""CLI `rag-ingest <ruta> --license <licencia>`: ingesta idempotente por hash (RF-01)."""

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy.exc import OperationalError

from rag_assistant.adapters.database import create_database_engine
from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.loaders.pdf import PdfLoader
from rag_assistant.adapters.loaders.text import TextLoader
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ingest import FileResult, IngestService, IngestStatus
from rag_assistant.config import get_settings
from rag_assistant.domain.errors import EmbeddingError

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-ingest",
        description="Ingesta documentos (Markdown, TXT o PDF) en la base vectorial.",
    )
    parser.add_argument("path", type=Path, help="archivo o carpeta a ingestar")
    parser.add_argument(
        "--license", required=True, help="licencia de los documentos, p. ej. CC-BY-4.0"
    )
    args = parser.parse_args(argv)
    if not args.path.exists():
        parser.error(f"no existe: {args.path}")

    settings = get_settings()
    if not settings.database_url:
        print("Falta RAG_DATABASE_URL: defínela en .env (ver .env.example).", file=sys.stderr)
        return EXIT_USAGE

    engine = create_database_engine(settings.database_url)
    try:
        service = IngestService(
            [MarkdownLoader(), TextLoader(), PdfLoader()],
            OllamaEmbeddings.from_url(
                settings.ollama_url,
                model=settings.embedding_model,
                batch_size=settings.embedding_batch_size,
            ),
            PgVectorStore(engine),
            on_result=print_result,
        )
        return run(service, args.path, license=args.license)
    finally:
        engine.dispose()


def run(service: IngestService, path: Path, *, license: str) -> int:
    started = time.perf_counter()
    try:
        report = service.ingest(path, license=license)
    except EmbeddingError as error:
        print(f"Error de embeddings: {error}", file=sys.stderr)
        return EXIT_FAILURE
    except OperationalError as error:
        print(f"La base de datos no responde: {error.orig}", file=sys.stderr)
        return EXIT_FAILURE

    totals = ", ".join(f"{report.count(status)} {status}" for status in IngestStatus)
    print(f"\n{len(report.results)} archivos en {time.perf_counter() - started:.1f} s: {totals}.")
    if report.count(IngestStatus.UNSUPPORTED):
        print(
            "Hay archivos con formato no soportado (se aceptan .md, .txt y .pdf).", file=sys.stderr
        )
        return EXIT_FAILURE
    return EXIT_OK


def print_result(result: FileResult) -> None:
    detail = f" ({result.chunks} fragmentos)" if result.status is IngestStatus.INGESTED else ""
    print(f"{result.status:<21} {result.source_path}{detail}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
