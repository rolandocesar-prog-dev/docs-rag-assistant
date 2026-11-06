"""Evaluación (RF-06): conjunto de preguntas, métricas de recuperación y veredictos del juez."""

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from rag_assistant.domain.answer import Answer
from rag_assistant.domain.retrieval import RetrievedChunk

OFF_TOPIC = "fuera_de_tema"
CATEGORIES = frozenset({"definicion", "procedimiento", "comparacion", "parafrasis", OFF_TOPIC})


@dataclass(frozen=True, slots=True)
class ExpectedSource:
    source_path: str
    section: str | None


@dataclass(frozen=True, slots=True)
class EvalItem:
    id: str
    category: str
    question: str
    expected: tuple[ExpectedSource, ...]
    reference: str | None

    @property
    def on_topic(self) -> bool:
        return self.category != OFF_TOPIC


def parse_dataset(lines: Iterable[str]) -> list[EvalItem]:
    """Lee el JSONL del conjunto; lanza `ValueError` con el número de línea si algo no cuadra."""
    items: list[EvalItem] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            item = EvalItem(
                id=str(raw["id"]),
                category=str(raw["category"]),
                question=str(raw["question"]),
                expected=tuple(
                    ExpectedSource(str(e["source_path"]), e["section"]) for e in raw["expected"]
                ),
                reference=raw["reference"],
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Línea {number} del conjunto inválida: {error}") from error
        if item.category not in CATEGORIES:
            raise ValueError(f"Línea {number}: categoría desconocida «{item.category}».")
        if item.on_topic != bool(item.expected):
            raise ValueError(
                f"Línea {number}: solo fuera de tema puede no tener fuentes esperadas."
            )
        items.append(item)
    return items


def is_relevant(result: RetrievedChunk, item: EvalItem, *, by_section: bool) -> bool:
    return any(
        result.source_path == expected.source_path
        and (not by_section or result.chunk.section == expected.section)
        for expected in item.expected
    )


def first_relevant_rank(
    results: Sequence[RetrievedChunk], item: EvalItem, *, by_section: bool
) -> int | None:
    """Posición (desde 1) del primer resultado relevante, o `None` si no hay ninguno."""
    return next(
        (
            rank
            for rank, r in enumerate(results, start=1)
            if is_relevant(r, item, by_section=by_section)
        ),
        None,
    )


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return _mean([1.0 if rank is not None and rank <= k else 0.0 for rank in ranks])


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    return _mean([1.0 / rank if rank is not None else 0.0 for rank in ranks])


def cites_expected_source(answer: Answer, item: EvalItem) -> bool:
    expected = {source.source_path for source in item.expected}
    return any(citation.source_path in expected for citation in answer.citations)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# --- Juez (fidelidad y corrección) -----------------------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "Eres un evaluador estricto de respuestas de un asistente sobre documentación de Kubernetes. "
    "Responde solo con un objeto JSON, sin texto adicional: "
    '{"fiel": true|false, "correcta": true|false}. '
    '"fiel" es true solo si todo lo que afirma la respuesta está respaldado por los fragmentos '
    'citados. "correcta" es true si la respuesta coincide en lo esencial con la respuesta de '
    "referencia, aunque use otras palabras o sea más breve."
)
_JSON_OBJECT = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Verdict:
    faithful: bool
    correct: bool


def build_judge_prompt(item: EvalItem, answer: Answer, fragments: Sequence[str]) -> str:
    cited = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(fragments, start=1))
    return (
        f"Pregunta: {item.question}\n\n"
        f"Fragmentos citados:\n{cited or '(ninguno)'}\n\n"
        f"Respuesta del asistente:\n{answer.text}\n\n"
        f"Respuesta de referencia:\n{item.reference}"
    )


def parse_verdict(text: str) -> Verdict | None:
    """El primer objeto JSON con `fiel` y `correcta` booleanos; `None` si no hay uno válido."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except ValueError:
        return None
    # El patrón solo captura `{…}`, así que lo leído siempre es un objeto.
    faithful, correct = raw.get("fiel"), raw.get("correcta")
    if not isinstance(faithful, bool) or not isinstance(correct, bool):
        return None
    return Verdict(faithful=faithful, correct=correct)
