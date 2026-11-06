"""Caso de uso de evaluación (RF-06): recuperación, respuestas y juez sobre el conjunto."""

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from rag_assistant.application.ask import AskService
from rag_assistant.domain.errors import EmbeddingError, LLMError
from rag_assistant.domain.evaluation import (
    JUDGE_SYSTEM_PROMPT,
    EvalItem,
    Verdict,
    build_judge_prompt,
    cites_expected_source,
    first_relevant_rank,
    hit_rate,
    mean_reciprocal_rank,
    parse_verdict,
)
from rag_assistant.ports.embedding_provider import EmbeddingProvider
from rag_assistant.ports.llm_provider import LLMProvider
from rag_assistant.ports.vector_store import VectorStore

DEFAULT_K = 10


@dataclass(frozen=True, slots=True)
class ItemResult:
    id: str
    category: str
    question: str
    on_topic: bool
    doc_rank: int | None
    section_rank: int | None
    found: bool
    cites_expected: bool
    verdict: Verdict | None
    seconds: float
    answer: str
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    n: int
    hit_at_1: float
    hit_at_answer_k: float  # @k de las respuestas: lo que de verdad recibe el LLM
    hit_at_k: float
    mrr: float


@dataclass(frozen=True, slots=True)
class AnswerMetrics:
    on_topic: int
    answered: int
    off_topic: int
    rejected: int
    cites_expected: int
    faithful: int  # sobre las respondidas; sin veredicto cuenta como no fiel
    correct: int  # sobre las del tema; sin respuesta o sin veredicto cuenta como incorrecta
    without_verdict: int
    errors: int
    latency_p50: float | None
    latency_p95: float | None


@dataclass(frozen=True, slots=True)
class EvalSummary:
    document: RetrievalMetrics
    section: RetrievalMetrics
    by_category: dict[str, RetrievalMetrics]
    answers: AnswerMetrics


@dataclass(slots=True)
class EvalReport:
    k: int
    answer_k: int
    judged: bool
    results: list[ItemResult] = field(default_factory=list)

    def summary(self) -> EvalSummary:
        on_topic = [r for r in self.results if r.on_topic]
        off_topic = [r for r in self.results if not r.on_topic]
        answered = [r for r in on_topic if r.found]
        timed = [r.seconds for r in self.results if r.error is None]
        return EvalSummary(
            document=self._retrieval([r.doc_rank for r in on_topic]),
            section=self._retrieval([r.section_rank for r in on_topic]),
            by_category={
                category: self._retrieval([r.doc_rank for r in on_topic if r.category == category])
                for category in sorted({r.category for r in on_topic})
            },
            answers=AnswerMetrics(
                on_topic=len(on_topic),
                answered=len(answered),
                off_topic=len(off_topic),
                rejected=sum(not r.found and r.error is None for r in off_topic),
                cites_expected=sum(r.cites_expected for r in on_topic),
                faithful=sum(bool(r.verdict and r.verdict.faithful) for r in answered),
                correct=sum(bool(r.verdict and r.verdict.correct) for r in on_topic),
                without_verdict=sum(r.verdict is None for r in answered) if self.judged else 0,
                errors=sum(r.error is not None for r in self.results),
                latency_p50=statistics.median(timed) if timed else None,
                latency_p95=_percentile(timed, 0.95),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "answer_k": self.answer_k,
            "judged": self.judged,
            "summary": asdict(self.summary()),
            "results": [asdict(result) for result in self.results],
        }

    def _retrieval(self, ranks: Sequence[int | None]) -> RetrievalMetrics:
        return RetrievalMetrics(
            n=len(ranks),
            hit_at_1=hit_rate(ranks, 1),
            hit_at_answer_k=hit_rate(ranks, self.answer_k),
            hit_at_k=hit_rate(ranks, self.k),
            mrr=mean_reciprocal_rank(ranks),
        )


class EvaluationService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        ask_service: AskService,
        judge: LLMProvider | None,
        *,
        k: int = DEFAULT_K,
        answer_k: int,
        clock: Callable[[], float] = time.perf_counter,
        on_result: Callable[[ItemResult], None] | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._ask = ask_service
        self._judge = judge
        self._k = k
        self._answer_k = answer_k
        self._clock = clock
        self._on_result = on_result

    def run(self, items: Sequence[EvalItem]) -> EvalReport:
        """Evalúa cada pregunta; un error del LLM queda en su resultado y no detiene el resto."""
        report = EvalReport(k=self._k, answer_k=self._answer_k, judged=self._judge is not None)
        for item in items:
            result = self._evaluate(item)
            report.results.append(result)
            if self._on_result is not None:
                self._on_result(result)
        return report

    def _evaluate(self, item: EvalItem) -> ItemResult:
        [vector] = self._embedder.embed([item.question])
        retrieved = self._store.search(vector, k=self._k)
        doc_rank = first_relevant_rank(retrieved, item, by_section=False) if item.on_topic else None
        section_rank = (
            first_relevant_rank(retrieved, item, by_section=True) if item.on_topic else None
        )

        start = self._clock()
        try:
            answer, context = self._ask.ask_with_context(item.question)
        except (LLMError, EmbeddingError) as error:
            return ItemResult(
                item.id, item.category, item.question, item.on_topic, doc_rank, section_rank,
                found=False, cites_expected=False, verdict=None,
                seconds=self._clock() - start, answer="", error=str(error),
            )  # fmt: skip
        seconds = self._clock() - start

        verdict = None
        if self._judge is not None and item.on_topic and answer.found:
            fragments = [context[citation.id - 1].chunk.content for citation in answer.citations]
            try:
                judged = self._judge.complete(
                    JUDGE_SYSTEM_PROMPT, build_judge_prompt(item, answer, fragments)
                )
                verdict = parse_verdict(judged.text)
            except LLMError:
                verdict = None
        return ItemResult(
            item.id, item.category, item.question, item.on_topic, doc_rank, section_rank,
            found=answer.found, cites_expected=cites_expected_source(answer, item),
            verdict=verdict, seconds=seconds, answer=answer.text,
        )  # fmt: skip


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]
