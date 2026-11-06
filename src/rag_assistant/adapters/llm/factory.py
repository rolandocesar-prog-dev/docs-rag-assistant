"""Elige el adaptador de LLM según `RAG_LLM_PROVIDER` (RF-04, design.md)."""

from rag_assistant.adapters.llm import anthropic as anthropic_llm
from rag_assistant.adapters.llm import ollama as ollama_llm
from rag_assistant.adapters.llm import openai_compatible
from rag_assistant.config import ConfigurationError, Settings
from rag_assistant.ports.llm_provider import LLMProvider

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"


def create_llm(settings: Settings) -> LLMProvider:
    """Lanza `ConfigurationError` si falta algo que el proveedor elegido necesita."""
    provider = settings.llm_provider
    if provider == "ollama":
        return ollama_llm.OllamaChat.from_url(
            settings.ollama_url,
            model=settings.llm_model or DEFAULT_OLLAMA_MODEL,
            max_tokens=settings.llm_max_tokens or ollama_llm.DEFAULT_MAX_TOKENS,
        )
    api_key = _required_api_key(settings)
    if provider == "openai_compatible":
        if not settings.llm_base_url or not settings.llm_model:
            raise ConfigurationError(
                "El proveedor openai_compatible necesita RAG_LLM_BASE_URL y RAG_LLM_MODEL "
                "(por ejemplo, https://api.openai.com/v1 y el nombre del modelo)."
            )
        return openai_compatible.OpenAICompatibleChat.from_url(
            settings.llm_base_url,
            api_key=api_key,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens or openai_compatible.DEFAULT_MAX_TOKENS,
        )
    return anthropic_llm.AnthropicChat.from_api_key(
        api_key,
        model=settings.llm_model or anthropic_llm.DEFAULT_MODEL,
        max_tokens=settings.llm_max_tokens or anthropic_llm.DEFAULT_MAX_TOKENS,
    )


def _required_api_key(settings: Settings) -> str:
    key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    if not key:
        raise ConfigurationError(
            f"Falta RAG_LLM_API_KEY: el proveedor {settings.llm_provider} la necesita "
            "(defínela en .env; nunca en el repositorio)."
        )
    return key
