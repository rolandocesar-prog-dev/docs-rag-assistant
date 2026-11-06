"""Adaptador de Anthropic con el SDK oficial y un transporte HTTP simulado (sin red ni claves)."""

import json

import anthropic
import httpx2
import pytest

from rag_assistant.adapters.llm.anthropic import AnthropicChat
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError
from rag_assistant.ports.llm_provider import LLMProvider

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def message(
    text: str, stop_reason: str = "end_turn", model: str = "claude-opus-5-5"
) -> dict[str, object]:
    return {
        "id": "msg_prueba",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "firma"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 30},
    }


def chat(
    handler: httpx2.MockTransport, *, model: str = "claude-opus-5-5", max_tokens: int = 4096
) -> LLMProvider:
    client = anthropic.Anthropic(
        api_key="clave-de-prueba",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(transport=handler),
    )
    return AnthropicChat(client, model=model, max_tokens=max_tokens)


def recorder(requests: list[httpx2.Request], payload: dict[str, object]) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=payload)

    return httpx2.MockTransport(handle)


def respond(status: int, payload: object | None = None) -> httpx2.MockTransport:
    body = payload or {"type": "error", "error": {"type": "error", "message": "algo falló"}}
    return httpx2.MockTransport(lambda _request: httpx2.Response(status, json=body))


def test_sends_system_user_effort_and_default_fallbacks() -> None:
    requests: list[httpx2.Request] = []

    completion = chat(recorder(requests, message("Un Pod agrupa contenedores [1]."))).complete(
        "Sistema.", "¿Qué es un Pod?"
    )

    assert completion == Completion("Un Pod agrupa contenedores [1].", truncated=False)
    [request] = requests
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 4096
    assert body["system"] == "Sistema."
    assert body["messages"] == [{"role": "user", "content": "¿Qué es un Pod?"}]
    assert body["output_config"] == {"effort": "low"}
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert request.headers["authorization"] == "Bearer clave-de-prueba"


def test_models_without_server_fallback_do_not_send_it() -> None:
    requests: list[httpx2.Request] = []

    chat(
        recorder(requests, message("Hola [1].", model="claude-haiku-5-5")), model="claude-haiku-5-5"
    ).complete("s", "u")

    body = json.loads(requests[0].content)
    assert "fallbacks" not in body
    assert "output_config" not in body
    assert FALLBACK_BETA not in requests[0].headers.get("anthropic-beta", "")


def test_max_tokens_stop_reason_marks_the_answer_as_truncated() -> None:
    requests: list[httpx2.Request] = []

    completion = chat(recorder(requests, message("[1] y [1]", "max_tokens"))).complete("s", "u")

    assert completion.truncated


def test_refusal_is_an_llm_error() -> None:
    with pytest.raises(LLMError, match="rechazó responder"):
        chat(recorder([], message("", "refusal"))).complete("s", "u")


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (401, "rechazó la clave"),
        (403, "no tiene permiso"),
        (404, "no tiene el modelo «claude-opus-5-5»"),
        (429, "límite de uso"),
        (500, "Anthropic respondió 500"),
    ],
    ids=["401", "403", "404", "429", "500"],
)
def test_api_errors_are_reported_as_llm_errors(status: int, message: str) -> None:
    with pytest.raises(LLMError, match=message):
        chat(respond(status)).complete("s", "u")


def test_unreachable_api_is_reported() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("conexión rechazada", request=request)

    with pytest.raises(LLMError, match="no responde"):
        chat(httpx2.MockTransport(refuse)).complete("s", "u")


def test_rejects_invalid_token_cap() -> None:
    with pytest.raises(ValueError, match="tope"):
        chat(respond(200), max_tokens=0)


def test_from_api_key_builds_a_client_with_that_key() -> None:
    llm = AnthropicChat.from_api_key("clave-de-prueba", model="claude-sonnet-5-5")

    assert llm._client.api_key == "clave-de-prueba"
    assert llm._model == "claude-sonnet-5-5"
