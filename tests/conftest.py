import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


class _TestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    test_database_url: str | None = None


@pytest.fixture(scope="session")
def test_database_url() -> str:
    """URL de la base de pruebas; sin RAG_TEST_DATABASE_URL, la prueba se omite (design.md)."""
    url = _TestSettings().test_database_url
    if not url:
        pytest.skip("RAG_TEST_DATABASE_URL no está definida")
    return url
