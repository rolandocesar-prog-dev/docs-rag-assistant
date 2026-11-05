from uuid import uuid4

import pytest

from fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore
from rag_assistant.application.ask import MAX_QUESTION_CHARS, AskService
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import InvalidQuestionError, LLMError
from rag_assistant.domain.prompting import NOT_FOUND, SYSTEM_PROMPT
from rag_assistant.domain.retrieval import RetrievedChunk


def result(title: str, content: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(uuid4(), 0, content, section=title),
        document_title=title,
        source_path=f"concepts/{title.lower()}.md",
        score=score,
    )


RESULTS = [
    result("Pods", "Un Pod agrupa contenedores.", 0.71),
    result("Nodos", "Un nodo es una máquina.", 0.45),
    result("Recetas", "Harina y huevos.", 0.31),
]


def service(
    llm: FakeLLM,
    results: list[RetrievedChunk] = RESULTS,
    embedder: FakeEmbedder | None = None,
    *,
    k: int = 4,
) -> tuple[AskService, InMemoryVectorStore]:
    store = InMemoryVectorStore(results)
    return AskService(embedder or FakeEmbedder(), store, llm, k=k, threshold=0.45), store


def test_answers_with_citations_using_only_relevant_fragments() -> None:
    llm = FakeLLM(Completion("Un Pod agrupa contenedores [1] y corre en un nodo [2]."))
    ask, store = service(llm)

    answer = ask.ask("  ¿Qué es un Pod?  ")

    assert answer.found
    assert [c.document for c in answer.citations] == ["Pods", "Nodos"]
    [(system, prompt)] = llm.calls
    assert system == SYSTEM_PROMPT
    assert "[1] Documento: Pods" in prompt and "[2] Documento: Nodos" in prompt
    assert "Recetas" not in prompt  # 0,31 no alcanza el umbral
    assert prompt.endswith("Pregunta: ¿Qué es un Pod?")
    assert store.search_ks == [4]


def test_without_relevant_fragments_it_does_not_call_the_llm() -> None:
    llm = FakeLLM(Completion("no debería usarse"))
    ask, _ = service(llm, [result("Recetas", "Harina y huevos.", 0.31)])

    answer = ask.ask("¿Cuál es la capital de Francia?")

    assert (answer.found, answer.text, answer.citations) == (False, NOT_FOUND, ())
    assert llm.calls == []


def test_empty_store_means_not_found() -> None:
    ask, _ = service(FakeLLM(Completion("x")), [])

    assert not ask.ask("¿Qué es un Pod?").found


def test_uncited_llm_answer_becomes_not_found() -> None:
    ask, _ = service(FakeLLM(Completion("Un Pod agrupa contenedores.")))

    assert not ask.ask("¿Qué es un Pod?").found


def test_truncated_llm_answer_is_an_error() -> None:
    ask, _ = service(FakeLLM(Completion("[1] y [3] son relevantes. [1] y", truncated=True)))

    with pytest.raises(LLMError, match="tope de tokens"):
        ask.ask("¿Qué es un Pod?")


@pytest.mark.parametrize(
    "question", ["", "   \n ", "x" * (MAX_QUESTION_CHARS + 1)], ids=["vacia", "espacios", "larga"]
)
def test_invalid_questions_are_rejected_before_searching(question: str) -> None:
    embedder = FakeEmbedder()
    ask, store = service(FakeLLM(Completion("x")), embedder=embedder)

    with pytest.raises(InvalidQuestionError):
        ask.ask(question)
    assert embedder.calls == []
    assert store.search_ks == []


def test_question_of_exactly_the_maximum_length_is_accepted() -> None:
    ask, _ = service(FakeLLM(Completion("Respuesta [1].")))

    assert ask.ask("x" * MAX_QUESTION_CHARS).found


@pytest.mark.parametrize(
    ("k", "threshold"),
    [(0, 0.45), (4, 1.5), (4, -1.5)],
    ids=["k-cero", "umbral-alto", "umbral-bajo"],
)
def test_rejects_invalid_configuration(k: int, threshold: float) -> None:
    with pytest.raises(ValueError):
        AskService(
            FakeEmbedder(),
            InMemoryVectorStore(),
            FakeLLM(Completion("x")),
            k=k,
            threshold=threshold,
        )
