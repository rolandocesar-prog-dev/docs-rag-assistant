"""Configuración de la aplicación, leída de variables de entorno (prefijo RAG_)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    app_name: str = "RAG Assistant"
    environment: str = "development"
    # Sin valor por defecto: las credenciales solo llegan por el entorno (RNF-02).
    database_url: str | None = None
    ollama_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "bge-m3"
    embedding_batch_size: int = 32
    llm_model: str = "qwen3.5:4b"
    llm_max_tokens: int = 400
    retrieval_k: int = 4
    relevance_threshold: float = 0.45


@lru_cache
def get_settings() -> Settings:
    return Settings()
