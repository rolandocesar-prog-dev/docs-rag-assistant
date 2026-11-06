"""Selección del LLM por `RAG_LLM_PROVIDER` (RF-04)."""

import pytest
from pydantic import SecretStr
from pydantic_settings import SettingsConfigDict

from rag_assistant.adapters.llm import anthropic as anthropic_llm
from rag_assistant.adapters.llm.anthropic import AnthropicChat
from rag_assistant.adapters.llm.factory import (
    DEFAULT_OLLAMA_MODEL,
    create_judge,
    create_llm,
    judge_label,
    llm_label,
)
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


def test_without_judge_provider_the_judge_is_the_answering_llm() -> None:
    values = settings(llm_model="qwen3.5:4b", llm_max_tokens=300)

    judge = create_judge(values)

    assert isinstance(judge, OllamaChat)
    assert (judge._model, judge._options["num_predict"]) == ("qwen3.5:4b", 300)
    assert judge_label(values) == llm_label(values) == "ollama/qwen3.5:4b"


def test_judge_has_its_own_provider_model_and_cap() -> None:
    values = settings(
        llm_provider="anthropic",
        llm_api_key=SecretStr("clave"),
        judge_provider="ollama",
        judge_model="qwen3.5:9b",
        judge_max_tokens=200,
    )

    judge = create_judge(values)

    assert isinstance(judge, OllamaChat)
    assert (judge._model, judge._options["num_predict"]) == ("qwen3.5:9b", 200)
    assert judge_label(values) == "ollama/qwen3.5:9b"
    assert llm_label(values) == f"anthropic/{anthropic_llm.DEFAULT_MODEL}"


def test_remote_judge_uses_its_own_key_and_names_judge_variables() -> None:
    remote = {"judge_provider": "openai_compatible", "llm_api_key": SecretStr("del-llm")}

    with pytest.raises(ConfigurationError, match="RAG_JUDGE_API_KEY"):
        create_judge(settings(**remote))
    with pytest.raises(ConfigurationError, match="RAG_JUDGE_BASE_URL y RAG_JUDGE_MODEL"):
        create_judge(settings(**remote, judge_api_key=SecretStr("clave")))
    judge = create_judge(
        settings(
            **remote,
            judge_api_key=SecretStr("clave"),
            judge_base_url="https://api.openai.com/v1",
            judge_model="gpt-x",
        )
    )
    assert isinstance(judge, OpenAICompatibleChat) and judge._model == "gpt-x"
