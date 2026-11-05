"""Conexión a PostgreSQL compartida por la CLI y la API."""

from sqlalchemy import Engine, create_engine

# Sin tiempo de espera, una base caída puede tardar minutos en dar error.
CONNECT_TIMEOUT_SECONDS = 10


def create_database_engine(url: str) -> Engine:
    return create_engine(url, connect_args={"connect_timeout": CONNECT_TIMEOUT_SECONDS})
