"""Llamada HTTP común a los adaptadores de Ollama, con errores explícitos (design.md)."""

import httpx2

from rag_assistant.adapters.http_json import post_json

# La primera llamada carga el modelo en memoria; en CPU puede tardar.
DEFAULT_TIMEOUT_SECONDS = 300.0


def new_client(base_url: str) -> httpx2.Client:
    return httpx2.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT_SECONDS)


def post_ollama(
    client: httpx2.Client,
    path: str,
    body: dict[str, object],
    *,
    model: str,
    error: type[Exception],
) -> object:
    """POST a Ollama; si falta el modelo, el error dice cómo descargarlo."""
    return post_json(
        client,
        path,
        body,
        error=error,
        provider="Ollama",
        not_found=(
            f"Ollama no tiene el modelo «{model}». Descárgalo con: ollama pull {model} "
            f"(con Ollama en Docker: docker compose exec ollama ollama pull {model})"
        ),
    )
