"""Contrato de `POST /ask` (design.md, sección 5), con dobles de los puertos."""

from collections.abc import Iterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore
from rag_assistant.api.dependencies import get_ask_service
from rag_assistant.application.ask import MAX_QUESTION_CHARS, AskService
from rag_assistant.config import get_settings
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.domain.prompting import NOT_FOUND
from rag_assistant.domain.retrieval import RetrievedChunk
from rag_assistant.main import create_app

POD = RetrievedChunk(
    chunk=Chunk(uuid4(), 0, "## ¿Qué es?\n\nUn Pod agrupa contenedores.", section="¿Qué es?"),
    document_title="Pods",
    source_path="concepts/workloads/pods/pod.md",
    score=0.72,
)


class FailingEmbedder(FakeEmbedder):
    def embed(self, texts: object) -> list[tuple[float, ...]]:
        raise EmbeddingError("Ollama no responde en http://127.0.0.1:11434")


class UnreachableStore(InMemoryVectorStore):
    def search(self, embedding: object, *, k: int) -> list[RetrievedChunk]:
        raise OperationalError("SELECT", {}, ConnectionRefusedError("conexión rechazada"))


@pytest.fixture
def app() -> Iterator[FastAPI]:
    application = create_app()
    yield application
    application.dependency_overrides.clear()


def client_with(app: FastAPI, service: AskService) -> TestClient:
    app.dependency_overrides[get_ask_service] = lambda: service
    return TestClient(app)


def service(
    completion: str = "Un Pod agrupa contenedores [1].",
    *,
    truncated: bool = False,
    results: list[RetrievedChunk] | None = None,
) -> AskService:
    return AskService(
        FakeEmbedder(),
        InMemoryVectorStore([POD] if results is None else results),
        FakeLLM(Completion(completion, truncated=truncated)),
    )


def test_returns_the_answer_with_its_citations(app: FastAPI) -> None:
    response = client_with(app, service()).post("/ask", json={"question": "¿Qué es un Pod?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Un Pod agrupa contenedores [1].",
        "citations": [
            {
                "id": 1,
                "document": "Pods",
                "section": "¿Qué es?",
                "page": None,
                "source_path": "concepts/workloads/pods/pod.md",
                "excerpt": "Un Pod agrupa contenedores.",
            }
        ],
        "found": True,
    }


def test_not_found_is_a_successful_response(app: FastAPI) -> None:
    response = client_with(app, service(results=[])).post(
        "/ask", json={"question": "¿Cuál es la capital de Francia?"}
    )

    assert response.status_code == 200
    assert response.json() == {"answer": NOT_FOUND, "citations": [], "found": False}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * (MAX_QUESTION_CHARS + 1)},
        {"question": 42},
        {"pregunta": "¿Qué es un Pod?"},
    ],
    ids=["sin-campo", "vacia", "espacios", "larga", "no-texto", "campo-equivocado"],
)
def test_invalid_questions_return_422(app: FastAPI, body: dict[str, object]) -> None:
    response = client_with(app, service()).post("/ask", json=body)

    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    ("ask_service", "status", "message"),
    [
        (service(truncated=True), 502, "tope de tokens"),
        (
            AskService(FailingEmbedder(), InMemoryVectorStore([POD]), FakeLLM(Completion("x"))),
            502,
            "Ollama no responde",
        ),
        (
            AskService(FakeEmbedder(), UnreachableStore(), FakeLLM(Completion("x"))),
            503,
            "La base de datos no responde",
        ),
    ],
    ids=["llm-cortado", "embeddings-caidos", "base-caida"],
)
def test_dependency_failures_return_explicit_errors(
    app: FastAPI, ask_service: AskService, status: int, message: str
) -> None:
    response = client_with(app, ask_service).post("/ask", json={"question": "¿Qué es un Pod?"})

    assert response.status_code == status
    assert message in response.json()["detail"]


def test_openapi_documents_the_endpoint(app: FastAPI) -> None:
    schema = TestClient(app).get("/openapi.json").json()

    assert "/ask" in schema["paths"]
    assert "AskResponse" in schema["components"]["schemas"]


@pytest.fixture
def fresh_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    get_settings.cache_clear()
    get_ask_service.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
    get_ask_service.cache_clear()


def test_real_service_is_built_from_the_settings(fresh_settings: pytest.MonkeyPatch) -> None:
    fresh_settings.setenv("RAG_DATABASE_URL", "postgresql+psycopg://rag:x@127.0.0.1:9/rag")

    assert isinstance(get_ask_service(), AskService)
    assert get_ask_service() is get_ask_service()


def test_missing_database_url_returns_503(
    app: FastAPI, fresh_settings: pytest.MonkeyPatch, tmp_path: object
) -> None:
    fresh_settings.chdir(str(tmp_path))  # sin .env en el directorio de trabajo
    fresh_settings.delenv("RAG_DATABASE_URL", raising=False)

    response = TestClient(app).post("/ask", json={"question": "¿Qué es un Pod?"})

    assert response.status_code == 503
    assert "RAG_DATABASE_URL" in response.json()["detail"]
