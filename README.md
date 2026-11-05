# Asistente RAG sobre documentación pública

[![CI](https://github.com/rolandocesar-prog-dev/docs-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/rolandocesar-prog-dev/docs-rag-assistant/actions/workflows/ci.yml)

Asistente de preguntas y respuestas que responde **solo con información de un conjunto de
documentos públicos** y **cita la fuente** de cada afirmación. Si la respuesta no está en los
documentos, lo dice en lugar de inventarla.

> 🚧 Proyecto en desarrollo. El avance está en [`specs/tasks.md`](./specs/tasks.md).

## Qué demuestra

- Recuperación aumentada por generación (RAG) implementada sin frameworks de orquestación.
- Arquitectura hexagonal: el proveedor del modelo (local o en la nube) se cambia por configuración.
- PostgreSQL + pgvector como base vectorial.
- Evaluación automática de la calidad de las respuestas.
- Tipado estricto, pruebas y CI desde el primer commit.

## Stack

Python 3.11+ · FastAPI · Pydantic · PostgreSQL + pgvector · Ollama / APIs de LLM · Pytest · Ruff ·
Mypy · Docker · GitHub Actions

## Cómo ejecutarlo

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env              # y cambia la contraseña de la base

docker compose up -d              # PostgreSQL + pgvector (5442) y Ollama (11434)
docker compose exec ollama ollama pull bge-m3   # modelo de embeddings, una sola vez (1,2 GB)
alembic upgrade head              # crea el esquema

uvicorn rag_assistant.main:app --reload
# http://localhost:8000/health  ·  documentación interactiva: http://localhost:8000/docs
```

## Calidad

```bash
ruff check . && ruff format --check .
mypy
pytest   # las pruebas de integración usan RAG_TEST_DATABASE_URL; sin ella, se omiten
```

## Cómo se desarrolla

El proyecto sigue **Spec Driven Development**: primero la especificación y después el código.

- [`specs/requirements.md`](./specs/requirements.md): qué debe hacer, con criterios de aceptación.
- [`specs/design.md`](./specs/design.md): arquitectura, modelo de datos y decisiones.
- [`specs/tasks.md`](./specs/tasks.md): plan por hitos.

## Licencia

Código bajo licencia [MIT](./LICENSE). Cada documento del corpus conserva su propia licencia,
registrada en `data/SOURCES.md`.
