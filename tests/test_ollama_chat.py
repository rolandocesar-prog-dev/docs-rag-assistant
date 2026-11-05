"""Adaptador de Ollama para el LLM con un transporte simulado (sin red)."""

import json
import re

import httpx2
import pytest

from rag_assistant.adapters.llm.ollama import OllamaChat
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError
from rag_assistant.ports.llm_provider import LLMProvider


def provider(transport: httpx2.MockTransport, *, max_tokens: int = 400) -> LLMProvider:
    client = httpx2.Client(base_url="http://ollama.test", transport=transport)
    return OllamaChat(client, model="qwen3.5:4b", max_tokens=max_tokens)


def respond(status: int, payload: object) -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda _request: httpx2.Response(status, json=payload))


def test_sends_system_and_user_messages_with_the_token_cap() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200, json={"message": {"content": "  Un Pod agrupa contenedores [1].\n"}}
        )

    completion = provider(httpx2.MockTransport(handle), max_tokens=123).complete(
        "Responde solo con el contexto.", "¿Qué es un Pod?"
    )

    assert completion == Completion("Un Pod agrupa contenedores [1].", truncated=False)
    [body] = requests
    assert body["model"] == "qwen3.5:4b"
    assert body["messages"] == [
        {"role": "system", "content": "Responde solo con el contexto."},
        {"role": "user", "content": "¿Qué es un Pod?"},
    ]
    assert (body["stream"], body["think"]) == (False, False)
    assert body["options"] == {"num_predict": 123, "num_ctx": 8192, "temperature": 0.1}


def test_marks_answers_that_hit_the_token_cap() -> None:
    payload = {
        "message": {"content": "[1] y [3] son relevantes. [1] y [3]"},
        "done_reason": "length",
    }

    assert provider(respond(200, payload)).complete("s", "u").truncated


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (404, {"error": "model not found"}, "ollama pull qwen3.5:4b"),
        (500, {"error": "boom"}, "500"),
        (200, {"message": {}}, "texto de la respuesta"),
        (200, ["no", "es", "un", "objeto"], "texto de la respuesta"),
    ],
    ids=["sin-modelo", "error-servidor", "sin-texto", "no-objeto"],
)
def test_errors_are_reported_as_llm_errors(status: int, payload: object, message: str) -> None:
    with pytest.raises(LLMError, match=message):
        provider(respond(status, payload)).complete("s", "u")


def test_non_json_response_is_an_llm_error() -> None:
    transport = httpx2.MockTransport(lambda _request: httpx2.Response(200, text="<html>"))

    with pytest.raises(LLMError, match="no es JSON"):
        provider(transport).complete("s", "u")


def test_rejects_invalid_token_cap() -> None:
    with pytest.raises(ValueError, match="tope"):
        provider(respond(200, {}), max_tokens=0)


def test_from_url_reports_an_unreachable_server() -> None:
    chat = OllamaChat.from_url("http://127.0.0.1:9", model="qwen3.5:4b")

    with pytest.raises(LLMError, match=re.escape("127.0.0.1:9")):
        chat.complete("s", "u")
