"""POST con JSON común a los adaptadores HTTP, con errores explícitos (design.md)."""

import httpx2


def post_json(
    client: httpx2.Client,
    path: str,
    body: dict[str, object],
    *,
    error: type[Exception],
    provider: str,
    not_found: str,
) -> object:
    """Envía `body` y devuelve el JSON; traduce cada fallo HTTP a `error` con un mensaje claro."""
    try:
        response = client.post(path, json=body)
    except httpx2.HTTPError as cause:
        raise error(f"{provider} no responde en {client.base_url}: {cause}") from cause
    if response.status_code in (httpx2.codes.UNAUTHORIZED, httpx2.codes.FORBIDDEN):
        raise error(f"{provider} rechazó la clave de API ({response.status_code}).")
    if response.status_code == httpx2.codes.NOT_FOUND:
        raise error(not_found)
    if response.is_error:
        raise error(f"{provider} respondió {response.status_code}: {response.text[:200]}")
    try:
        return response.json()
    except ValueError as cause:
        raise error(f"{provider} devolvió una respuesta que no es JSON: {cause}") from cause
