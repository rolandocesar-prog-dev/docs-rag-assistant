"""Modelo de lenguaje con Ollama (`POST /api/chat`), con tope de tokens (design.md)."""

import httpx2

from rag_assistant.adapters.ollama_http import new_client, post_json
from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError

DEFAULT_MAX_TOKENS = 400
DEFAULT_CONTEXT_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.1


class OllamaChat:
    def __init__(
        self,
        client: httpx2.Client,
        *,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("El tope de tokens debe ser mayor o igual a 1.")
        self._client = client
        self._model = model
        self._options = {
            "num_predict": max_tokens,
            "num_ctx": context_tokens,
            "temperature": temperature,
        }

    @classmethod
    def from_url(
        cls, base_url: str, *, model: str, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> "OllamaChat":
        return cls(new_client(base_url), model=model, max_tokens=max_tokens)

    def complete(self, system: str, user: str) -> Completion:
        payload = post_json(
            self._client,
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                # Sin «razonamiento» previo: más rápido y basta para responder con el contexto.
                "think": False,
                "options": self._options,
            },
            model=self._model,
            error=LLMError,
        )
        message = payload.get("message") if isinstance(payload, dict) else None
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise LLMError("Ollama no devolvió el texto de la respuesta.")
        truncated = isinstance(payload, dict) and payload.get("done_reason") == "length"
        return Completion(text=text.strip(), truncated=truncated)
