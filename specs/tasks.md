# Tareas — Asistente RAG sobre documentación pública

Cada tarea referencia el requisito que cumple. Marca `[x]` al terminar y haz un commit por tarea.

## M0 · Base del proyecto

- [x] Estructura `src/`, `pyproject.toml`, Ruff, Mypy estricto y Pytest.
- [x] Endpoint `GET /health` con prueba.
- [x] CI en GitHub Actions (lint, formato, tipos, pruebas).
- [x] Especificaciones SDD: requisitos, diseño y tareas.

## M1 · Ingesta (RF-01)

- [x] Elegir el corpus de demostración y registrar sus licencias en `data/SOURCES.md`.
- [x] Entidades `Document` y `Chunk` y el puerto `DocumentLoader`.
- [x] Cargadores de Markdown y TXT con pruebas; PDF con `pypdf`.
- [ ] Divisor de fragmentos por encabezado y tamaño, con pruebas de bordes.
- [ ] `docker-compose.yml` con PostgreSQL + pgvector y migración inicial.
- [ ] Adaptador `VectorStore` (pgvector) con prueba de integración.
- [ ] Puerto `EmbeddingProvider` y adaptador Ollama.
- [ ] CLI `rag-ingest <carpeta>` idempotente por hash.

## M2 · Preguntas y citas (RF-02, RF-03)

- [ ] `AskService` con umbral de relevancia y respuesta "no encontré información".
- [ ] Puerto `LLMProvider` y adaptador Ollama.
- [ ] Prompt con fragmentos numerados y validación de citas.
- [ ] Endpoint `POST /ask` con validación (422) y pruebas de contrato.

## M3 · Proveedores e interfaz (RF-04, RF-05)

- [ ] Adaptadores compatibles con OpenAI y con Anthropic; selección por `RAG_LLM_PROVIDER`.
- [ ] Error explícito al iniciar si falta la clave del proveedor.
- [ ] Página web de chat con indicador de carga y citas desplegables.

## M4 · Evaluación (RF-06)

- [ ] 20+ preguntas de prueba en `eval/dataset.jsonl`.
- [ ] Comando `rag-eval` con hit rate@k, MRR y fidelidad; reporte en `eval/reports/`.

## M5 · Entrega

- [ ] `Dockerfile` de la API y `docker compose up` de punta a punta (RNF-05).
- [ ] Logs estructurados con id de petición (RNF-04).
- [ ] Despliegue público o video de demostración.
- [ ] README final con GIF, arquitectura y resultados de evaluación.
- [ ] Agregar el proyecto, con enlaces, a la sección Proyectos del CV.
