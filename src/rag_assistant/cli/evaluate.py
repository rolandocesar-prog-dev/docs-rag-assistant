"""CLI `rag-eval`: evalúa el asistente con `eval/dataset.jsonl` y guarda el reporte (RF-06)."""

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from sqlalchemy.exc import OperationalError

from rag_assistant.adapters.database import create_database_engine
from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.adapters.llm.factory import create_llm, llm_label
from rag_assistant.adapters.vectorstore.pgvector import PgVectorStore
from rag_assistant.application.ask import AskService
from rag_assistant.application.evaluate import DEFAULT_K, EvalReport, EvaluationService, ItemResult
from rag_assistant.config import ConfigurationError, get_settings
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.evaluation import parse_dataset

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-eval", description="Evalúa el asistente RAG.")
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    parser.add_argument("--salida", type=Path, default=Path("eval/reports"))
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="fragmentos para recuperación")
    parser.add_argument("--sin-juez", action="store_true", help="omite fidelidad y corrección")
    args = parser.parse_args(argv)

    settings = get_settings()
    try:
        items = parse_dataset(args.dataset.read_text(encoding="utf-8").splitlines())
        llm = create_llm(settings)
    except (OSError, ValueError, ConfigurationError) as error:
        print(f"No se puede evaluar: {error}", file=sys.stderr)
        return EXIT_USAGE
    if not settings.database_url:
        print("Falta RAG_DATABASE_URL: defínela en .env (ver .env.example).", file=sys.stderr)
        return EXIT_USAGE

    engine = create_database_engine(settings.database_url)
    embedder = OllamaEmbeddings.from_url(settings.ollama_url, model=settings.embedding_model)
    store = PgVectorStore(engine)
    ask = AskService(
        embedder,
        store,
        llm,
        k=settings.retrieval_k,
        threshold=settings.relevance_threshold,
        hybrid=settings.hybrid_search,
    )
    service = EvaluationService(
        embedder,
        store,
        ask,
        None if args.sin_juez else llm,
        k=args.k,
        answer_k=settings.retrieval_k,
        hybrid=settings.hybrid_search,
        on_result=print_result,
    )
    try:
        report = service.run(items)
    except (EmbeddingError, OperationalError) as error:
        print(f"La evaluación se detuvo: {error}", file=sys.stderr)
        return EXIT_FAILURE
    finally:
        engine.dispose()

    label = llm_label(settings)
    paths = write_report(report, args.salida, label=label, when=datetime.now())
    print(f"\n{render_summary(report)}\nReporte: {paths[1]}")
    return EXIT_OK


def print_result(result: ItemResult) -> None:
    rank = "-" if result.doc_rank is None else str(result.doc_rank)
    state = "error" if result.error else ("responde" if result.found else "no encontró")
    print(f"{result.id} {result.category:<14} pos. {rank:>2}  {state:<12} {result.seconds:5.1f} s")


def write_report(
    report: EvalReport, folder: Path, *, label: str, when: datetime
) -> tuple[Path, Path]:
    """Escribe `<fecha>-<modelo>.json` y `.md` en `folder` y devuelve sus rutas."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{when:%Y-%m-%d-%H%M}-{re.sub(r'[^a-zA-Z0-9]+', '-', label).strip('-')}"
    json_path, md_path = folder / f"{stem}.json", folder / f"{stem}.md"
    data = {"date": when.isoformat(timespec="seconds"), "llm": label} | report.to_dict()
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report, label=label, when=when), encoding="utf-8")
    return json_path, md_path


def render_summary(report: EvalReport) -> str:
    summary = report.summary()
    document, answers = summary.document, summary.answers
    return (
        f"Recuperación (documento): hit@{report.answer_k} {_pct(document.hit_at_answer_k)}, "
        f"hit@{report.k} {_pct(document.hit_at_k)}, MRR {document.mrr:.2f}\n"
        f"Respuestas: {answers.answered}/{answers.on_topic} del tema, "
        f"rechazo {answers.rejected}/{answers.off_topic} fuera de tema"
    )


def render_markdown(report: EvalReport, *, label: str, when: datetime) -> str:
    summary = report.summary()
    answers = summary.answers
    k, answer_k = report.k, report.answer_k
    judge = label if report.judged else "sin juez"

    lines = [
        f"# Evaluación del asistente — {when:%Y-%m-%d %H:%M}",
        "",
        f"- Modelo: `{label}` · Juez: `{judge}` · Preguntas: {answers.on_topic} del tema y "
        f"{answers.off_topic} fuera de tema · Recuperación con k = {k}; respuestas con "
        f"k = {answer_k}.",
        "",
        "## Recuperación (preguntas del tema)",
        "",
        f"| Nivel | hit@1 | hit@{answer_k} | hit@{k} | MRR@{k} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, m in (("Documento", summary.document), ("Sección", summary.section)):
        lines.append(
            f"| {name} | {_pct(m.hit_at_1)} | {_pct(m.hit_at_answer_k)} | {_pct(m.hit_at_k)} "
            f"| {m.mrr:.2f} |"
        )
    lines += [
        "",
        "Por categoría (documento):",
        "",
        f"| Categoría | n | hit@1 | hit@{answer_k} | MRR@{k} |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category, m in summary.by_category.items():
        lines.append(
            f"| {category} | {m.n} | {_pct(m.hit_at_1)} | {_pct(m.hit_at_answer_k)} | {m.mrr:.2f} |"
        )
    lines += [
        "",
        "## Respuestas",
        "",
        "| Métrica | Resultado |",
        "| --- | --- |",
        f"| Responde las preguntas del tema | {_ratio(answers.answered, answers.on_topic)} |",
        f"| Rechaza las de fuera de tema | {_ratio(answers.rejected, answers.off_topic)} |",
        f"| Cita la fuente esperada | {_ratio(answers.cites_expected, answers.on_topic)} |",
    ]
    if report.judged:
        lines += [
            f"| Fidelidad (juez, sobre las respondidas) | "
            f"{_ratio(answers.faithful, answers.answered)} |",
            f"| Corrección (juez, sobre las del tema) | "
            f"{_ratio(answers.correct, answers.on_topic)} |",
            f"| Sin veredicto del juez | {answers.without_verdict} |",
        ]
    lines += [
        f"| Errores del modelo | {answers.errors} |",
        f"| Latencia p50 / p95 | {_seconds(answers.latency_p50)} / "
        f"{_seconds(answers.latency_p95)} |",
        "",
        "## Detalle por pregunta",
        "",
        "| id | Categoría | Pregunta | Pos. doc. | Pos. sección | Respuesta | Cita fuente "
        "| Fiel | Correcta | Tiempo |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        state = "error" if r.error else ("sí" if r.found else "no encontró")
        verdict = r.verdict
        # Una pregunta del tema sin respuesta cuenta como incorrecta en el resumen.
        correct = verdict.correct if verdict else (False if not r.found else None)
        lines.append(
            f"| {r.id} | {r.category} | {r.question} | {_rank(r.doc_rank, r.on_topic)} | "
            f"{_rank(r.section_rank, r.on_topic)} | {state} | {_yes(r.cites_expected, r.on_topic)} "
            f"| {_yes(verdict.faithful if verdict else None, r.found and report.judged)} "
            f"| {_yes(correct, r.on_topic and report.judged)} "
            f"| {r.seconds:.1f} s |"
        )
    lines += [
        "",
        "El juez es el mismo modelo que responde, así que sus veredictos son orientativos: sirven",
        "para comparar versiones del sistema, no como medida absoluta de calidad.",
        "",
    ]
    return "\n".join(lines)


def _pct(value: float) -> str:
    return f"{value * 100:.0f} %"


def _ratio(part: int, total: int) -> str:
    return f"{part}/{total} ({part / total * 100:.0f} %)" if total else "—"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} s"


def _rank(rank: int | None, applies: bool) -> str:
    if not applies:
        return "—"
    return "no aparece" if rank is None else str(rank)


def _yes(value: bool | None, applies: bool) -> str:
    if not applies:
        return "—"
    return "sin veredicto" if value is None else ("sí" if value else "no")


if __name__ == "__main__":
    sys.exit(main())
