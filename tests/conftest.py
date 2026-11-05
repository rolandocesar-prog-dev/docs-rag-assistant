from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine, text

ROOT = Path(__file__).resolve().parents[1]


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


@pytest.fixture(scope="session")
def alembic_config(test_database_url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.set_main_option("sqlalchemy.url", test_database_url.replace("%", "%%"))
    return config


@pytest.fixture
def engine(test_database_url: str, alembic_config: Config) -> Iterator[Engine]:
    """Base de pruebas con el esquema al día y sin datos de pruebas anteriores."""
    command.upgrade(alembic_config, "head")
    engine = create_engine(test_database_url)
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE documents CASCADE"))
    yield engine
    engine.dispose()
