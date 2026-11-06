"""Adaptador compatible con OpenAI con un transporte simulado (sin red ni claves)."""

import json
import re

import httpx2
import pytest

from rag_assistant.adapters.llm.openai_compatible import OpenAICompatibleChat
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError
from rag_assistant.ports.llm_provider import LLMProvider

BASE_URL = "https://llm.test/v1/"


def provider(transport: httpx2.MockTransport) -> LLMProvider:
    client = httpx2.Client(base_url=BASE_URL, transport=transport)
    return OpenAICompatibleChat(client, model="gpt-x", max_tokens=200)


def respond(status: int, payload: object) -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda _request: httpx2.Response(status, json=payload))


def answer(content: object, finish_reason: str = "stop") -> dict[str, object]:
    return {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}


def test_posts_chat_completions_with_system_user_and_token_cap() -> None:
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, json=answer(" Un Pod agrupa contenedores [1]. "))

    completion = provider(httpx2.MockTransport(handle)).complete("Sistema.", "¿Qué es un Pod?")

    assert completion == Completion("Un Pod agrupa contenedores [1].", truncated=False)
    [request] = requests
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    body = json.loads(request.content)
    assert body == {
        "model": "gpt-x",
        "messages": [
            {"role": "system", "content": "Sistema."},
            {"role": "user", "content": "¿Qué es un Pod?"},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
    }


def test_from_url_sends_the_api_key_as_bearer_token() -> None:
    chat = OpenAICompatibleChat.from_url(BASE_URL, api_key="clave-de-prueba", model="gpt-x")

    assert chat._client.headers["Authorization"] == "Bearer clave-de-prueba"


def test_length_finish_reason_marks_the_answer_as_truncated() -> None:
    assert provider(respond(200, answer("[1] y [1] y", "length"))).complete("s", "u").truncated


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (401, {"error": {"message": "invalid key"}}, "rechazó la clave"),
        (403, {"error": {"message": "forbidden"}}, "rechazó la clave"),
        (404, {"error": {"message": "model not found"}}, "no encontró el modelo «gpt-x»"),
        (429, {"error": {"message": "rate limit"}}, "429"),
        (200, {"choices": []}, "texto de la respuesta"),
        (200, answer(None), "texto de la respuesta"),
        (200, ["no", "es", "objeto"], "texto de la respuesta"),
    ],
    ids=["401", "403", "404", "429", "sin-opciones", "sin-texto", "no-objeto"],
)
def test_errors_are_reported_as_llm_errors(status: int, payload: object, message: str) -> None:
    with pytest.raises(LLMError, match=message):
        provider(respond(status, payload)).complete("s", "u")


def test_unreachable_service_is_reported() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("conexión rechazada", request=request)

    with pytest.raises(LLMError, match=re.escape("no responde en https://llm.test/v1/")):
        provider(httpx2.MockTransport(refuse)).complete("s", "u")


def test_rejects_invalid_token_cap() -> None:
    with pytest.raises(ValueError, match="tope"):
        OpenAICompatibleChat(httpx2.Client(), model="gpt-x", max_tokens=0)
