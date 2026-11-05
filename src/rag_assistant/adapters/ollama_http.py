"""Llamada HTTP común a los adaptadores de Ollama, con errores explícitos (design.md)."""

import httpx2

# La primera llamada carga el modelo en memoria; en CPU puede tardar.
DEFAULT_TIMEOUT_SECONDS = 300.0


def new_client(base_url: str) -> httpx2.Client:
    return httpx2.Client(base_url=base_url, timeout=DEFAULT_TIMEOUT_SECONDS)


def post_json(
    client: httpx2.Client,
    path: str,
    body: dict[str, object],
    *,
    model: str,
    error: type[Exception],
) -> object:
    """POST a Ollama; traduce los fallos HTTP a `error` con un mensaje accionable."""
    try:
        response = client.post(path, json=body)
    except httpx2.HTTPError as cause:
        raise error(f"Ollama no responde en {client.base_url}: {cause}") from cause
    if response.status_code == httpx2.codes.NOT_FOUND:
        raise error(
            f"Ollama no tiene el modelo «{model}». Descárgalo con: ollama pull {model} "
            f"(con Ollama en Docker: docker compose exec ollama ollama pull {model})"
        )
    if response.is_error:
        raise error(f"Ollama respondió {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as cause:
        raise error(f"Ollama devolvió una respuesta que no es JSON: {cause}") from cause
