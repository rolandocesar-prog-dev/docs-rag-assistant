"""LLM con la API de Anthropic, mediante el SDK oficial (RF-04, design.md)."""

from typing import Final

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaOutputConfigParam

from rag_assistant.domain.completion import Completion
from rag_assistant.domain.errors import LLMError

DEFAULT_MODEL = "claude-opus-5-5"
# El razonamiento interno cuenta dentro de max_tokens: el tope debe dejarle margen.
DEFAULT_MAX_TOKENS = 4096
# Respuestas breves de chat: el esfuerzo bajo alcanza y responde más rápido.
DEFAULT_EFFORT: Final = "low"
# Modelos que admiten el reintento automático en otro modelo si los filtros rechazan por error
# (`fallbacks: "default"`, beta server-side-fallback-2026-07-01).
FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)


class AnthropicChat:
    def __init__(
        self,
        client: anthropic.Anthropic,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("El tope de tokens debe ser mayor o igual a 1.")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @classmethod
    def from_api_key(
        cls, api_key: str, *, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS
    ) -> "AnthropicChat":
        return cls(anthropic.Anthropic(api_key=api_key), model=model, max_tokens=max_tokens)

    def complete(self, system: str, user: str) -> Completion:
        messages: list[BetaMessageParam] = [{"role": "user", "content": user}]
        # El esfuerzo y el reintento en otro modelo solo se envían a los modelos que los admiten.
        current = self._model in FALLBACK_MODELS
        output_config: BetaOutputConfigParam = {"effort": DEFAULT_EFFORT}
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=messages,
                output_config=output_config if current else anthropic.omit,
                betas=["server-side-fallback-2026-07-01"] if current else anthropic.omit,
                fallbacks="default" if current else anthropic.omit,
            )
        except anthropic.AuthenticationError as error:
            raise LLMError("Anthropic rechazó la clave de API (RAG_LLM_API_KEY).") from error
        except anthropic.PermissionDeniedError as error:
            raise LLMError(f"La clave de API no tiene permiso: {error.message}") from error
        except anthropic.NotFoundError as error:
            raise LLMError(f"Anthropic no tiene el modelo «{self._model}».") from error
        except anthropic.RateLimitError as error:
            raise LLMError("Se alcanzó el límite de uso de Anthropic; prueba más tarde.") from error
        except anthropic.APIStatusError as error:
            raise LLMError(f"Anthropic respondió {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise LLMError(f"Anthropic no responde: {error}") from error

        if response.stop_reason == "refusal":
            raise LLMError("El modelo rechazó responder esta pregunta.")
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(text=text.strip(), truncated=response.stop_reason == "max_tokens")
