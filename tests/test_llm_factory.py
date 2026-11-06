"""Selección del LLM por `RAG_LLM_PROVIDER` (RF-04)."""

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from rag_assistant.adapters.llm import anthropic as anthropic_llm
from rag_assistant.adapters.llm.anthropic import AnthropicChat
from rag_assistant.adapters.llm.factory import DEFAULT_OLLAMA_MODEL, create_llm
from rag_assistant.adapters.llm.ollama import DEFAULT_MAX_TOKENS, OllamaChat
from rag_assistant.adapters.llm.openai_compatible import OpenAICompatibleChat
from rag_assistant.config import ConfigurationError, Settings


class IsolatedSettings(Settings):
    """No lee `.env`: cada prueba fija sus propios valores."""

    model_config = SettingsConfigDict(env_file=None, env_prefix="RAG_", extra="ignore")


def settings(**values: object) -> Settings:
    return IsolatedSettings.model_validate(values)


def test_ollama_is_the_default_with_its_own_model_and_cap() -> None:
    llm = create_llm(settings())

    assert isinstance(llm, OllamaChat)
    assert llm._model == DEFAULT_OLLAMA_MODEL
    assert llm._options["num_predict"] == DEFAULT_MAX_TOKENS


def test_anthropic_uses_its_defaults_and_the_key() -> None:
    llm = create_llm(settings(llm_provider="anthropic", llm_api_key=SecretStr("clave")))

    assert isinstance(llm, AnthropicChat)
    assert (llm._model, llm._max_tokens) == (
        anthropic_llm.DEFAULT_MODEL,
        anthropic_llm.DEFAULT_MAX_TOKENS,
    )


def test_openai_compatible_uses_base_url_model_and_cap() -> None:
    llm = create_llm(
        settings(
            llm_provider="openai_compatible",
            llm_api_key=SecretStr("clave"),
            llm_base_url="https://api.openai.com/v1",
            llm_model="gpt-x",
            llm_max_tokens=300,
        )
    )

    assert isinstance(llm, OpenAICompatibleChat)
    assert (llm._model, llm._max_tokens) == ("gpt-x", 300)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"llm_provider": "anthropic"}, "RAG_LLM_API_KEY"),
        ({"llm_provider": "anthropic", "llm_api_key": SecretStr("   ")}, "RAG_LLM_API_KEY"),
        ({"llm_provider": "openai_compatible"}, "RAG_LLM_API_KEY"),
        (
            {"llm_provider": "openai_compatible", "llm_api_key": SecretStr("clave")},
            "RAG_LLM_BASE_URL",
        ),
    ],
    ids=["anthropic-sin-clave", "clave-en-blanco", "openai-sin-clave", "openai-sin-url"],
)
def test_missing_settings_are_configuration_errors(values: dict[str, object], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        create_llm(settings(**values))


def test_unknown_provider_is_rejected_when_reading_settings() -> None:
    with pytest.raises(ValueError, match="llm_provider"):
        settings(llm_provider="otro")
