"""CLI `rag-eval`: reporte en JSON y Markdown, y errores de uso (RF-06)."""

import json
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

import pytest

from rag_assistant.application.evaluate import EvalReport, ItemResult
from rag_assistant.cli import evaluate as cli
from rag_assistant.config import get_settings
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.evaluation import Verdict

WHEN = datetime(2026, 11, 6, 10, 30)


def report(*, judged: bool = True) -> EvalReport:
    results = [
        ItemResult("q01", "definicion", "¿Qué es un Pod?", True, 1, 2, True, True,
                   Verdict(True, True) if judged else None, 2.0, "Un Pod [1]."),
        ItemResult("q02", "parafrasis", "¿Contraseña?", True, None, None, True, False,
                   Verdict(False, False) if judged else None, 3.0, "Otra cosa [1]."),
        ItemResult("q03", "fuera_de_tema", "¿Francia?", False, None, None, False, False,
                   None, 0.1, "No encontré información."),
        ItemResult("q04", "definicion", "¿Nodo?", True, 4, None, False, False, None, 9.0, "",
                   error="tope de tokens"),
    ]  # fmt: skip
    return EvalReport(k=10, answer_k=4, judged=judged, results=results)


def test_writes_json_and_markdown_with_the_same_name(tmp_path: Path) -> None:
    json_path, md_path = cli.write_report(
        report(), tmp_path / "reports", label="ollama/qwen3.5:4b", when=WHEN
    )

    assert json_path.name == "2026-11-06-1030-ollama-qwen3-5-4b.json"
    assert md_path.name == "2026-11-06-1030-ollama-qwen3-5-4b.md"
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert (data["llm"], data["date"]) == ("ollama/qwen3.5:4b", "2026-11-06T10:30:00")
    assert data["summary"]["answers"]["errors"] == 1
    markdown = md_path.read_text(encoding="utf-8")
    for expected in (
        "| Documento | 33 % | 67 % | 67 % | 0.42 |",
        "| Responde las preguntas del tema | 2/3 (67 %) |",
        "| Rechaza las de fuera de tema | 1/1 (100 %) |",
        "| Fidelidad (juez, sobre las respondidas) | 1/2 (50 %) |",
        "| q02 | parafrasis | ¿Contraseña? | no aparece | no aparece | sí | no | no | no | 3.0 s |",
        "| q04 | definicion | ¿Nodo? | 4 | no aparece | error | no | — | no | 9.0 s |",
        "| q03 | fuera_de_tema | ¿Francia? | — | — | no encontró | — | — | — | 0.1 s |",
    ):
        assert expected in markdown


def test_markdown_without_judge_omits_judge_metrics() -> None:
    markdown = cli.render_markdown(report(judged=False), label="ollama/qwen3.5:4b", when=WHEN)

    assert "Fidelidad" not in markdown
    assert "Juez: `sin juez`" in markdown


def test_summary_line_and_progress(capsys: pytest.CaptureFixture[str]) -> None:
    sample = report()
    cli.print_result(sample.results[0])
    cli.print_result(sample.results[3])

    output = capsys.readouterr().out
    assert "q01 definicion" in output and "pos.  1" in output and "responde" in output
    assert "error" in output
    assert "hit@4 67 %" in cli.render_summary(sample)


@pytest.fixture
def fresh_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[pytest.MonkeyPatch]:
    monkeypatch.chdir(tmp_path)  # sin .env
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_usage_errors_return_exit_code_2(
    fresh_settings: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "category": "fuera_de_tema", "question": "¿x?", "expected": [],
                    "reference": None}) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    fresh_settings.delenv("RAG_DATABASE_URL", raising=False)

    assert cli.main(["--dataset", str(tmp_path / "no-existe.jsonl")]) == cli.EXIT_USAGE
    assert cli.main(["--dataset", str(dataset)]) == cli.EXIT_USAGE
    fresh_settings.setenv("RAG_LLM_PROVIDER", "anthropic")
    get_settings.cache_clear()
    assert cli.main(["--dataset", str(dataset)]) == cli.EXIT_USAGE

    errors = capsys.readouterr().err
    assert "no-existe.jsonl" in errors
    assert "RAG_DATABASE_URL" in errors
    assert "RAG_LLM_API_KEY" in errors


def test_stops_with_exit_code_1_if_embeddings_fail(
    fresh_settings: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingEvaluation:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def run(self, items: object) -> EvalReport:
            raise EmbeddingError("Ollama no responde")

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        json.dumps({"id": "q1", "category": "fuera_de_tema", "question": "¿x?", "expected": [],
                    "reference": None}) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    fresh_settings.setenv("RAG_DATABASE_URL", "postgresql+psycopg://rag:x@127.0.0.1:9/rag")
    fresh_settings.setattr(cli, "EvaluationService", FailingEvaluation)

    assert cli.main(["--dataset", str(dataset), "--salida", str(tmp_path)]) == cli.EXIT_FAILURE
    assert "La evaluación se detuvo: Ollama no responde" in capsys.readouterr().err
