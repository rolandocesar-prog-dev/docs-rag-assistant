# Imagen de la API y de los comandos (specs/design.md, «Todo con docker compose up»).

# --- build: instala el paquete y sus dependencias en un entorno virtual ----------------------
FROM python:3.12-slim AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# --- runtime: solo el entorno, las migraciones y el corpus -----------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH

RUN useradd --create-home --uid 10001 rag
COPY --from=build /opt/venv /opt/venv

WORKDIR /app
COPY alembic.ini ./
COPY migrations ./migrations
COPY data/corpus ./data/corpus

USER rag
EXPOSE 8000
CMD ["uvicorn", "rag_assistant.main:app", "--host", "0.0.0.0", "--port", "8000"]
