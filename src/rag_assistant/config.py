"""Configuración de la aplicación, leída de variables de entorno (prefijo RAG_)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    app_name: str = "RAG Assistant"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
