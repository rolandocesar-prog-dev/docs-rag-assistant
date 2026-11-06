"""EvaluationService con dobles: rangos, respuestas, juez y errores (RF-06)."""

import json
from collections.abc import Iterator
from uuid import uuid4

from fakes import FakeEmbedder, InMemoryVectorStore, ScriptedLLM
from rag_assistant.application.ask import AskService
from rag_assistant.application.evaluate import EvaluationService
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError
from rag_assistant.domain.evaluation import EvalItem, ExpectedSource, Verdict
from rag_assistant.domain.prompting import NOT_FOUND
from rag_assistant.domain.retrieval import RetrievedChunk

POD = RetrievedChunk(
    Chunk(uuid4(), 0, "Un Pod agrupa contenedores.", section="¿Qué es?"),
    "Pods",
    "concepts/pod.md",
    0.7,
)
NODE = RetrievedChunk(
    Chunk(uuid4(), 0, "Un nodo es una máquina.", section=None), "Nodos", "concepts/nodo.md", 0.6
)
ON_TOPIC = EvalItem(
    "q01",
    "definicion",
    "¿Qué es un Pod?",
    (ExpectedSource("concepts/pod.md", "¿Qué es?"),),
    "Agrupa contenedores.",
)
OFF_TOPIC = EvalItem("q02", "fuera_de_tema", "¿Capital de Francia?", (), None)


def ticks() -> Iterator[float]:
    value = 0.0
    while True:
        yield value
        value += 1.5


def service(llm: ScriptedLLM, *, judge: bool = True) -> EvaluationService:
    store = InMemoryVectorStore([NODE, POD])
    clock = ticks()
    return EvaluationService(
        FakeEmbedder(),
        store,
        AskService(FakeEmbedder(), store, llm),
        llm if judge else None,
        k=10,
        answer_k=4,
        clock=lambda: next(clock),
    )


def test_measures_retrieval_answers_and_judge() -> None:
    llm = ScriptedLLM(
        Completion("Un Pod agrupa contenedores [2]."),
        Completion('{"fiel": true, "correcta": true}'),
        Completion(NOT_FOUND),
    )
    notified: list[str] = []
    evaluation = service(llm)
    evaluation._on_result = lambda r: notified.append(r.id)

    report = evaluation.run([ON_TOPIC, OFF_TOPIC])

    on, off = report.results
    assert (on.doc_rank, on.section_rank, on.found, on.cites_expected) == (2, 2, True, True)
    assert on.verdict == Verdict(faithful=True, correct=True)
    assert on.seconds == 1.5
    assert (off.doc_rank, off.found, off.verdict) == (None, False, None)
    assert notified == ["q01", "q02"]
    judge_prompt = llm.calls[1][1]
    # El fragmento citado, completo y con el mismo número que usa la respuesta.
    assert "[2] Un Pod agrupa contenedores." in judge_prompt
    assert "[1]" not in judge_prompt

    summary = report.summary()
    assert summary.document.hit_at_1 == 0.0 and summary.document.hit_at_answer_k == 1.0
    assert summary.document.mrr == 0.5
    assert summary.by_category["definicion"].n == 1
    answers = summary.answers
    assert (answers.answered, answers.rejected, answers.cites_expected) == (1, 1, 1)
    assert (answers.faithful, answers.correct, answers.without_verdict, answers.errors) == (
        1,
        1,
        0,
        0,
    )
    assert answers.latency_p50 == 1.5
    assert json.loads(json.dumps(report.to_dict()))["summary"]["answers"]["answered"] == 1


def test_llm_errors_are_recorded_without_stopping() -> None:
    llm = ScriptedLLM(Completion("[1] y [1]", truncated=True), Completion(NOT_FOUND))

    report = service(llm).run([ON_TOPIC, OFF_TOPIC])

    first, second = report.results
    assert first.error and "tope de tokens" in first.error
    assert not first.found and second.error is None
    assert report.summary().answers.errors == 1


def test_judge_failures_leave_the_item_without_verdict() -> None:
    llm = ScriptedLLM(Completion("Un Pod agrupa contenedores [2]."), LLMError("caído"))

    [result] = service(llm).run([ON_TOPIC]).results

    assert result.found and result.verdict is None


def test_without_judge_no_verdicts_are_requested() -> None:
    llm = ScriptedLLM(Completion("Un Pod agrupa contenedores [2]."))

    report = service(llm, judge=False).run([ON_TOPIC])

    assert not report.judged and report.results[0].verdict is None
    assert len(llm.calls) == 1
    assert report.summary().answers.without_verdict == 0


def test_latency_is_unknown_when_every_answer_failed() -> None:
    llm = ScriptedLLM(LLMError("caído"))

    answers = service(llm).run([ON_TOPIC]).summary().answers

    assert (answers.errors, answers.latency_p50, answers.latency_p95) == (1, None, None)


def test_hybrid_evaluation_searches_with_the_question_text() -> None:
    store = InMemoryVectorStore([NODE, POD])
    llm = ScriptedLLM(Completion(NOT_FOUND))
    evaluation = EvaluationService(
        FakeEmbedder(), store, AskService(FakeEmbedder(), store, llm, hybrid=True), None,
        k=10, answer_k=4, hybrid=True,
    )  # fmt: skip

    evaluation.run([OFF_TOPIC])

    assert store.search_texts == ["¿Capital de Francia?", "¿Capital de Francia?"]
