"""Caso de uso de pregunta (RF-02, RF-03): buscar, filtrar por relevancia, generar y citar."""

from rag_assistant.domain.answer import Answer
from rag_assistant.domain.errors import InvalidQuestionError, LLMError
from rag_assistant.domain.prompting import (
    SYSTEM_PROMPT,
    build_prompt,
    not_found_answer,
    parse_answer,
)
from rag_assistant.domain.retrieval import RetrievedChunk
from rag_assistant.ports.embedding_provider import EmbeddingProvider
from rag_assistant.ports.llm_provider import LLMProvider
from rag_assistant.ports.vector_store import VectorStore

MAX_QUESTION_CHARS = 1000
DEFAULT_K = 4
# Medido con el corpus: fuera de tema ≈ 0,3; del tema entre 0,6 y 0,7 (design.md).
DEFAULT_RELEVANCE_THRESHOLD = 0.45


class AskService:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        *,
        k: int = DEFAULT_K,
        threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
    ) -> None:
        if k < 1:
            raise ValueError("k debe ser mayor o igual a 1.")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("El umbral es una similitud de coseno: debe estar entre -1 y 1.")
        self._embedder = embedder
        self._store = store
        self._llm = llm
        self._k = k
        self._threshold = threshold

    def ask(self, question: str) -> Answer:
        answer, _ = self.ask_with_context(question)
        return answer

    def ask_with_context(self, question: str) -> tuple[Answer, list[RetrievedChunk]]:
        """La respuesta y los fragmentos relevantes que recibió el LLM (los usa la evaluación)."""
        question = _validate(question)
        [vector] = self._embedder.embed([question])
        relevant = [
            item for item in self._store.search(vector, k=self._k) if item.score >= self._threshold
        ]
        if not relevant:
            return not_found_answer(), []
        completion = self._llm.complete(SYSTEM_PROMPT, build_prompt(question, relevant))
        if completion.truncated:
            raise LLMError("El modelo llegó al tope de tokens sin terminar la respuesta.")
        return parse_answer(completion.text, relevant), relevant


def _validate(question: str) -> str:
    question = question.strip()
    if not question:
        raise InvalidQuestionError("La pregunta no puede estar vacía.")
    if len(question) > MAX_QUESTION_CHARS:
        raise InvalidQuestionError(
            f"La pregunta supera los {MAX_QUESTION_CHARS} caracteres ({len(question)})."
        )
    return question
