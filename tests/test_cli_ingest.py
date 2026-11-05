from pathlib import Path

import pytest
from sqlalchemy.exc import OperationalError

from fakes import FakeEmbedder, InMemoryVectorStore
from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.application.ingest import IngestService
from rag_assistant.cli import ingest as cli
from rag_assistant.config import get_settings


class UnreachableStore(InMemoryVectorStore):
    def content_hash(self, source_path: str) -> str | None:
        raise OperationalError("SELECT", {}, ConnectionRefusedError("conexión rechazada"))


def make_service(embedder: FakeEmbedder) -> IngestService:
    return IngestService(
        [MarkdownLoader()], embedder, InMemoryVectorStore(), on_result=cli.print_result
    )


def test_prints_one_line_per_file_and_a_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pods.md").write_text("# Pods\n\nUn Pod agrupa contenedores.", encoding="utf-8")

    code = cli.run(make_service(FakeEmbedder()), tmp_path / "pods.md", license="MIT")

    output = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "ingestado" in output and "pods.md (1 fragmentos)" in output
    assert "1 archivos" in output and "1 ingestado" in output


def test_unsupported_files_make_the_exit_code_non_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "informe.docx").write_bytes(b"binario")

    code = cli.run(make_service(FakeEmbedder()), tmp_path, license="MIT")

    assert code == cli.EXIT_FAILURE
    captured = capsys.readouterr()
    assert "formato no soportado" in captured.out
    assert ".md, .txt y .pdf" in captured.err


def test_embedding_errors_are_reported_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pods.md").write_text("# Pods\n\nTexto.", encoding="utf-8")

    code = cli.run(make_service(FakeEmbedder(fail=True)), tmp_path, license="MIT")

    assert code == cli.EXIT_FAILURE
    assert "Error de embeddings" in capsys.readouterr().err


def test_unreachable_database_is_reported_with_exit_code_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "pods.md").write_text("# Pods\n\nTexto.", encoding="utf-8")
    service = IngestService([MarkdownLoader()], FakeEmbedder(), UnreachableStore())

    code = cli.run(service, tmp_path, license="MIT")

    assert code == cli.EXIT_FAILURE
    assert "La base de datos no responde" in capsys.readouterr().err


def test_missing_database_url_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)  # sin .env en el directorio de trabajo
    monkeypatch.delenv("RAG_DATABASE_URL", raising=False)
    get_settings.cache_clear()

    code = cli.main([str(tmp_path), "--license", "MIT"])

    get_settings.cache_clear()
    assert code == cli.EXIT_USAGE
    assert "RAG_DATABASE_URL" in capsys.readouterr().err


def test_requires_an_existing_path_and_a_license(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as missing_license:
        cli.main([str(tmp_path)])
    with pytest.raises(SystemExit) as missing_path:
        cli.main([str(tmp_path / "no-existe"), "--license", "MIT"])

    assert missing_license.value.code == cli.EXIT_USAGE
    assert missing_path.value.code == cli.EXIT_USAGE
