"""Adaptador de Ollama con un transporte simulado (sin red): lotes, orden y errores (RF-04)."""

import json
import re

import httpx2
import pytest

from rag_assistant.adapters.embeddings.ollama import OllamaEmbeddings
from rag_assistant.domain.errors import EmbeddingError
from rag_assistant.ports.embedding_provider import EmbeddingProvider

DIMS = 4


def fake_vector(text: str) -> list[float]:
    """Vector determinista que identifica al texto: su longitud en la primera componente."""
    return [float(len(text)), 0.0, 0.0, 1.0]


def provider(
    handler: httpx2.MockTransport, *, batch_size: int = 2, dimensions: int = DIMS
) -> EmbeddingProvider:
    client = httpx2.Client(base_url="http://ollama.test", transport=handler)
    return OllamaEmbeddings(client, model="bge-m3", batch_size=batch_size, dimensions=dimensions)


def echo_handler(requests: list[dict[str, object]]) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(200, json={"embeddings": [fake_vector(t) for t in body["input"]]})

    return httpx2.MockTransport(handle)


def respond(status: int, payload: object = None) -> httpx2.MockTransport:
    return httpx2.MockTransport(lambda _request: httpx2.Response(status, json=payload))


def test_embeds_in_batches_and_keeps_the_order() -> None:
    requests: list[dict[str, object]] = []
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = provider(echo_handler(requests), batch_size=2).embed(texts)

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [request["input"] for request in requests] == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert all(request["model"] == "bge-m3" for request in requests)


def test_no_texts_means_no_requests() -> None:
    requests: list[dict[str, object]] = []

    assert provider(echo_handler(requests)).embed([]) == []
    assert requests == []


def test_missing_model_explains_how_to_download_it() -> None:
    with pytest.raises(EmbeddingError, match="ollama pull bge-m3"):
        provider(respond(404, {"error": "model not found"})).embed(["texto"])


def test_server_error_includes_the_status() -> None:
    with pytest.raises(EmbeddingError, match="500"):
        provider(respond(500, {"error": "boom"})).embed(["texto"])


def test_unreachable_server_is_reported() -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("conexión rechazada", request=request)

    with pytest.raises(EmbeddingError, match="no responde"):
        provider(httpx2.MockTransport(refuse)).embed(["texto"])


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"embeddings": [[0.1, 0.2]]}, "2 dimensiones"),
        ({"embeddings": []}, "no devolvió 1"),
        ({"otra": "cosa"}, "no devolvió 1"),
        (["no", "es", "un", "objeto"], "no devolvió 1"),
        ({"embeddings": ["texto"]}, "\\? dimensiones"),
    ],
    ids=["dimension", "cantidad", "sin-campo", "no-objeto", "no-vector"],
)
def test_invalid_responses_stop_the_ingestion(payload: object, message: str) -> None:
    with pytest.raises(EmbeddingError, match=message):
        provider(respond(200, payload)).embed(["texto"])


def test_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError, match="lote"):
        provider(respond(200), batch_size=0)


def test_from_url_builds_a_client_for_that_server() -> None:
    embeddings = OllamaEmbeddings.from_url("http://127.0.0.1:9", model="bge-m3")

    with pytest.raises(EmbeddingError, match=re.escape("127.0.0.1:9")):
        embeddings.embed(["texto"])
