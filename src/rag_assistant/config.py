"""Configuración de la aplicación, leída de variables de entorno (prefijo RAG_)."""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["ollama", "openai_compatible", "anthropic"]


class ConfigurationError(RuntimeError):
    """Falta o es inválida una variable de entorno necesaria (RF-04)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    app_name: str = "RAG Assistant"
    environment: str = "development"
    # Sin valor por defecto: las credenciales solo llegan por el entorno (RNF-02).
    database_url: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "bge-m3"
    embedding_batch_size: int = 32
    # LLM (RF-04): sin modelo ni tope, se usan los valores por defecto del proveedor.
    llm_provider: LLMProviderName = "ollama"
    llm_model: str | None = None
    llm_max_tokens: int | None = None
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    retrieval_k: int = 4
    # Búsqueda híbrida (vector + texto completo en español); false = solo vector.
    hybrid_search: bool = True
    relevance_threshold: float = 0.45


@lru_cache
def get_settings() -> Settings:
    return Settings()
