"""Logs JSON e id de petición (RNF-04, design.md)."""

import io
import json
import logging
import sys
from collections.abc import Callable, Iterator
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fakes import FakeEmbedder, FakeLLM, InMemoryVectorStore, ScriptedLLM
from rag_assistant.api.dependencies import get_ask_service
from rag_assistant.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from rag_assistant.application.ask import AskService
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError
from rag_assistant.domain.retrieval import RetrievedChunk
from rag_assistant.logs import (
    JsonFormatter,
    RequestIdFilter,
    _AppHandler,
    configure_logging,
    request_id_var,
)
from rag_assistant.main import create_app
from rag_assistant.ports.llm_provider import LLMProvider

QUESTION = "¿Qué es un Pod? Me llamo Ana"
POD = RetrievedChunk(Chunk(uuid4(), 0, "Un Pod agrupa contenedores."), "Pods", "pods.md", 0.7)


Lines = Callable[[], list[dict[str, object]]]


@pytest.fixture
def logged() -> Iterator[Lines]:
    """Devuelve las líneas JSON que escribió la app (con el filtro del id de petición)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(JsonFormatter())
    app_logger = logging.getLogger("rag_assistant")
    app_logger.addHandler(handler)
    previous = app_logger.level
    app_logger.setLevel(logging.DEBUG)
    yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    app_logger.removeHandler(handler)
    app_logger.setLevel(previous)


def client(llm: LLMProvider) -> TestClient:
    app = create_app()
    store = InMemoryVectorStore([POD])
    app.dependency_overrides[get_ask_service] = lambda: AskService(FakeEmbedder(), store, llm)
    return TestClient(app)


# --- Formato ----------------------------------------------------------------------------------


def test_json_line_has_time_level_logger_message_request_id_and_extras() -> None:
    record = logging.LogRecord(
        "rag_assistant.x", logging.INFO, "f.py", 1, "Hola %s", ("Ana",), None
    )
    record.status = 200
    record.color_message = "[1mHola[0m"
    token = request_id_var.set("abc123")
    try:
        RequestIdFilter().filter(record)
    finally:
        request_id_var.reset(token)

    entry = json.loads(JsonFormatter().format(record))

    assert entry["level"] == "INFO" and entry["logger"] == "rag_assistant.x"
    assert (entry["message"], entry["request_id"], entry["status"]) == ("Hola Ana", "abc123", 200)
    assert str(entry["time"]).endswith("+00:00")
    assert not {"args", "msg", "color_message"} & entry.keys()


def test_json_includes_the_exception_and_omits_request_id_outside_requests() -> None:
    try:
        raise RuntimeError("falló")
    except RuntimeError:
        record = logging.LogRecord("x", logging.ERROR, "f.py", 1, "Error", None, sys.exc_info())
    RequestIdFilter().filter(record)

    entry = json.loads(JsonFormatter().format(record))

    assert "request_id" not in entry
    assert "RuntimeError: falló" in str(entry["exception"])


def test_configure_logging_replaces_only_its_own_handler() -> None:
    root = logging.getLogger()
    other = logging.NullHandler()
    root.addHandler(other)
    try:
        configure_logging("text", "DEBUG")
        configure_logging("json", "INFO")

        own = [h for h in root.handlers if isinstance(h, _AppHandler)]
        assert len(own) == 1 and isinstance(own[0].formatter, JsonFormatter)
        assert other in root.handlers and root.level == logging.INFO
        assert logging.getLogger("uvicorn.error").propagate
        assert not logging.getLogger("uvicorn.access").propagate
    finally:
        root.removeHandler(other)


def test_text_format_shows_a_dash_outside_requests(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("text", "INFO")
    try:
        logging.getLogger("rag_assistant.prueba").info("Arranca")
        assert "INFO    rag_assistant.prueba [-] Arranca" in capsys.readouterr().out
    finally:
        configure_logging("json", "INFO")


# --- Id de petición y una línea por petición -----------------------------------------------------


def test_each_request_gets_an_id_in_the_header_and_in_its_logs(
    logged: Lines,
) -> None:
    response = client(FakeLLM(Completion("Un Pod agrupa contenedores [1]."))).post(
        "/ask", json={"question": QUESTION}
    )

    request_id = response.headers[REQUEST_ID_HEADER]
    assert len(request_id) == 32
    entries = logged()
    answered = next(e for e in entries if e["message"] == "Pregunta respondida")
    access = next(e for e in entries if e["message"] == "Petición")
    assert answered["request_id"] == access["request_id"] == request_id
    assert (answered["found"], answered["citations"]) == (True, 1)
    assert answered["question_chars"] == len(QUESTION)
    assert (access["method"], access["path"], access["status"]) == ("POST", "/ask", 200)
    assert isinstance(access["duration_ms"], float)
    assert all("Ana" not in json.dumps(e, ensure_ascii=False) for e in entries)


@pytest.mark.parametrize(
    ("incoming", "kept"),
    [("proxy-1.abc_Z", True), ("con espacios", False), ("x" * 65, False)],
    ids=["valido", "espacios", "largo"],
)
def test_incoming_request_id_is_kept_only_if_safe(incoming: str, kept: bool) -> None:
    response = client(FakeLLM(Completion("x"))).get(
        "/health", headers={REQUEST_ID_HEADER: incoming}
    )

    assert (response.headers[REQUEST_ID_HEADER] == incoming) is kept


def test_health_is_logged_at_debug_and_upstream_errors_as_warnings(
    logged: Lines,
) -> None:
    http = client(ScriptedLLM(LLMError("Ollama no responde")))
    http.get("/health")
    http.post("/ask", json={"question": "¿Qué es un Pod?"})

    entries = logged()
    health = next(e for e in entries if e.get("path") == "/health")
    assert health["level"] == "DEBUG"
    warning = next(e for e in entries if e["level"] == "WARNING")
    assert (warning["error"], warning["status"]) == ("LLMError", 502)
    assert warning["request_id"] is not None


def test_unexpected_errors_are_logged_with_their_trace_and_status_500(
    logged: Lines,
) -> None:
    app = FastAPI()

    @app.get("/falla")
    def fail() -> None:
        raise RuntimeError("inesperado")

    app.add_middleware(RequestContextMiddleware)

    response = TestClient(app, raise_server_exceptions=False).get("/falla")

    assert response.status_code == 500
    entries = logged()
    error = next(e for e in entries if e["level"] == "ERROR")
    assert "RuntimeError: inesperado" in str(error["exception"])
    access = next(e for e in entries if e["message"] == "Petición")
    assert access["status"] == 500 and access["request_id"] == error["request_id"]


def test_lifespan_scope_passes_through_without_request_id() -> None:
    with TestClient(create_app()) as http:  # el lifespan pasa por el middleware sin id
        assert http.get("/health").status_code == 200
    assert request_id_var.get() is None
