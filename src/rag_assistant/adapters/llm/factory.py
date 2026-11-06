"""Elige el adaptador de LLM según `RAG_LLM_PROVIDER` y el juez según `RAG_JUDGE_*` (design.md)."""

from dataclasses import dataclass

from pydantic import SecretStr

from rag_assistant.adapters.llm import anthropic as anthropic_llm
from rag_assistant.adapters.llm import ollama as ollama_llm
from rag_assistant.adapters.llm import openai_compatible
from rag_assistant.config import ConfigurationError, LLMProviderName, Settings
from rag_assistant.ports.llm_provider import LLMProvider

DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Lo que necesita un proveedor. Ollama usa `RAG_OLLAMA_URL` tanto para el LLM como para el
    juez; `prefix` nombra las variables del grupo en los mensajes de error."""

    provider: LLMProviderName
    ollama_url: str
    model: str | None = None
    max_tokens: int | None = None
    api_key: SecretStr | None = None
    base_url: str | None = None
    prefix: str = "RAG_LLM_"


def validate_llm_settings(settings: Settings) -> None:
    """Revisa la configuración del LLM sin conectarse; lanza `ConfigurationError`."""
    _validate(_llm_config(settings))


def create_llm(settings: Settings) -> LLMProvider:
    """Lanza `ConfigurationError` si falta algo que el proveedor elegido necesita."""
    return _create(_llm_config(settings))


def create_judge(settings: Settings) -> LLMProvider:
    """El juez de `rag-eval`: el de `RAG_JUDGE_*` o, sin `RAG_JUDGE_PROVIDER`, el mismo LLM."""
    return _create(_judge_config(settings))


def llm_label(settings: Settings) -> str:
    """Proveedor y modelo efectivos, p. ej. «ollama/qwen3.5:4b» (para reportes)."""
    return _label(_llm_config(settings))


def judge_label(settings: Settings) -> str:
    return _label(_judge_config(settings))


def _llm_config(settings: Settings) -> LLMConfig:
    return LLMConfig(
        provider=settings.llm_provider,
        ollama_url=settings.ollama_url,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )


def _judge_config(settings: Settings) -> LLMConfig:
    if settings.judge_provider is None:
        return _llm_config(settings)
    return LLMConfig(
        provider=settings.judge_provider,
        ollama_url=settings.ollama_url,
        model=settings.judge_model,
        max_tokens=settings.judge_max_tokens,
        api_key=settings.judge_api_key,
        base_url=settings.judge_base_url,
        prefix="RAG_JUDGE_",
    )


def _validate(config: LLMConfig) -> None:
    if config.provider == "ollama":
        return
    _required_api_key(config)
    if config.provider == "openai_compatible":
        _openai_target(config)


def _create(config: LLMConfig) -> LLMProvider:
    if config.provider == "ollama":
        return ollama_llm.OllamaChat.from_url(
            config.ollama_url,
            model=config.model or DEFAULT_OLLAMA_MODEL,
            max_tokens=config.max_tokens or ollama_llm.DEFAULT_MAX_TOKENS,
        )
    api_key = _required_api_key(config)
    if config.provider == "openai_compatible":
        base_url, model = _openai_target(config)
        return openai_compatible.OpenAICompatibleChat.from_url(
            base_url,
            api_key=api_key,
            model=model,
            max_tokens=config.max_tokens or openai_compatible.DEFAULT_MAX_TOKENS,
        )
    return anthropic_llm.AnthropicChat.from_api_key(
        api_key,
        model=config.model or anthropic_llm.DEFAULT_MODEL,
        max_tokens=config.max_tokens or anthropic_llm.DEFAULT_MAX_TOKENS,
    )


def _label(config: LLMConfig) -> str:
    default = {
        "ollama": DEFAULT_OLLAMA_MODEL,
        "anthropic": anthropic_llm.DEFAULT_MODEL,
        "openai_compatible": "?",
    }[config.provider]
    return f"{config.provider}/{config.model or default}"


def _openai_target(config: LLMConfig) -> tuple[str, str]:
    if not config.base_url or not config.model:
        raise ConfigurationError(
            f"El proveedor openai_compatible necesita {config.prefix}BASE_URL y "
            f"{config.prefix}MODEL (por ejemplo, https://api.openai.com/v1 y el nombre del modelo)."
        )
    return config.base_url, config.model


def _required_api_key(config: LLMConfig) -> str:
    key = config.api_key.get_secret_value().strip() if config.api_key else ""
    if not key:
        raise ConfigurationError(
            f"Falta {config.prefix}API_KEY: el proveedor {config.provider} la necesita "
            "(defínela en .env; nunca en el repositorio)."
        )
    return key
