"""Entorno de Alembic: toma la URL de RAG_DATABASE_URL y aplica migraciones de SQL explícito."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from rag_assistant.config import get_settings

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def database_url() -> str:
    """La URL de `sqlalchemy.url` (la fijan las pruebas) o, si no hay, la de RAG_DATABASE_URL."""
    url = config.get_main_option("sqlalchemy.url") or get_settings().database_url
    if not url:
        raise RuntimeError("Falta RAG_DATABASE_URL: defínela en .env (ver .env.example).")
    return url


def run_migrations_offline() -> None:
    """Genera el SQL sin conectarse (`alembic upgrade head --sql`)."""
    context.configure(url=database_url(), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
