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

docker compose up -d              # PostgreSQL + pgvector en 127.0.0.1:5442
alembic upgrade head              # crea el esquema

# Ollama en 127.0.0.1:11434. Con GPU, instálalo en el equipo (https://ollama.com/download);
# si no, úsalo en Docker: docker compose --profile ollama up -d
ollama pull bge-m3                # embeddings, una sola vez (1,2 GB)
ollama pull qwen3.5:4b            # modelo de lenguaje, una sola vez (3,3 GB)
# En Docker: docker compose exec ollama ollama pull bge-m3  (y lo mismo con qwen3.5:4b)

# Ingesta del corpus (≈ 5 min la primera vez en CPU; después solo procesa lo que cambió)
rag-ingest data/corpus/kubernetes/concepts --license CC-BY-4.0

uvicorn rag_assistant.main:app --reload
# Chat: http://localhost:8000/  ·  API: http://localhost:8000/docs  ·  salud: /health
```

## Preguntar

En **http://localhost:8000/** hay una página de chat: muestra un indicador mientras busca y, debajo
de cada respuesta, sus fuentes como tarjetas desplegables; las marcas `[n]` del texto abren la
fuente correspondiente. También se puede usar la API directamente:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es un StatefulSet?"}'
```

```json
{
  "answer": "Un StatefulSet es el objeto de la API workload diseñado para gestionar aplicaciones con estado, proporcionando identificaciones de red estables y almacenamiento persistente [1]. …",
  "citations": [
    {
      "id": 1,
      "document": "StatefulSets",
      "section": null,
      "page": null,
      "source_path": "concepts/workloads/controllers/statefulset.md",
      "excerpt": "Un StatefulSet es el objeto de la API workload que se usa para gestionar aplicaciones con estado…"
    }
  ],
  "found": true
}
```

Si la documentación no cubre la pregunta, responde `found: false` sin inventar nada. Errores: `422`
(pregunta vacía o de más de 1 000 caracteres), `502` (falló el modelo) y `503` (la base no
responde).

## Cambiar de modelo de lenguaje

Por defecto responde un modelo local (Ollama). Para usar un servicio remoto basta con el `.env`,
sin tocar código:

```bash
# Anthropic (SDK oficial; por defecto claude-opus-5-5)
RAG_LLM_PROVIDER=anthropic
RAG_LLM_API_KEY=...

# Cualquier API compatible con OpenAI
RAG_LLM_PROVIDER=openai_compatible
RAG_LLM_BASE_URL=https://api.openai.com/v1
RAG_LLM_MODEL=...
RAG_LLM_API_KEY=...
```

Los embeddings siguen en Ollama con `bge-m3`, así que la base no hay que volver a ingestarla.

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
