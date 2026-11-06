"""RF-04: si falta la configuración del proveedor de LLM, la API no arranca."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from rag_assistant.config import ConfigurationError, get_settings
from rag_assistant.main import create_app


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in ("RAG_LLM_API_KEY", "RAG_LLM_BASE_URL", "RAG_LLM_MODEL"):
        monkeypatch.setenv(name, "")
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("variables", "missing"),
    [
        ({"RAG_LLM_PROVIDER": "anthropic"}, "RAG_LLM_API_KEY"),
        ({"RAG_LLM_PROVIDER": "openai_compatible"}, "RAG_LLM_API_KEY"),
        ({"RAG_LLM_PROVIDER": "openai_compatible", "RAG_LLM_API_KEY": "clave"}, "RAG_LLM_BASE_URL"),
    ],
    ids=["anthropic-sin-clave", "openai-sin-clave", "openai-sin-url"],
)
def test_app_does_not_start_without_the_provider_settings(
    environment: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    variables: dict[str, str],
    missing: str,
) -> None:
    for name, value in variables.items():
        environment.setenv(name, value)

    with pytest.raises(ConfigurationError, match=missing), TestClient(create_app()):
        pass
    assert missing in caplog.text


@pytest.mark.parametrize(
    "variables",
    [
        {"RAG_LLM_PROVIDER": "ollama"},
        {"RAG_LLM_PROVIDER": "anthropic", "RAG_LLM_API_KEY": "clave"},
        {
            "RAG_LLM_PROVIDER": "openai_compatible",
            "RAG_LLM_API_KEY": "clave",
            "RAG_LLM_BASE_URL": "https://api.openai.com/v1",
            "RAG_LLM_MODEL": "gpt-x",
        },
    ],
    ids=["ollama", "anthropic", "openai"],
)
def test_app_starts_with_a_complete_configuration(
    environment: pytest.MonkeyPatch, variables: dict[str, str]
) -> None:
    for name, value in variables.items():
        environment.setenv(name, value)

    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
