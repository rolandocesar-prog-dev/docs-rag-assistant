"""LLM con cualquier API compatible con OpenAI (`POST /chat/completions`), por HTTP (RF-04)."""

import httpx2

from rag_assistant.adapters.http_json import post_json
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT_SECONDS = 120.0


class OpenAICompatibleChat:
    def __init__(
        self,
        client: httpx2.Client,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("El tope de tokens debe ser mayor o igual a 1.")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    @classmethod
    def from_url(
        cls, base_url: str, *, api_key: str, model: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> "OpenAICompatibleChat":
        client = httpx2.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        return cls(client, model=model, max_tokens=max_tokens)

    def complete(self, system: str, user: str) -> Completion:
        payload = post_json(
            self._client,
            "chat/completions",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
            error=LLMError,
            provider="El servicio compatible con OpenAI",
            not_found=(
                f"El servicio compatible con OpenAI no encontró el modelo «{self._model}» "
                f"o la ruta chat/completions en {self._client.base_url}."
            ),
        )
        choices = payload.get("choices") if isinstance(payload, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise LLMError("El servicio no devolvió el texto de la respuesta.")
        truncated = isinstance(choice, dict) and choice.get("finish_reason") == "length"
        return Completion(text=text.strip(), truncated=truncated)
