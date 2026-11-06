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
- [x] Divisor de fragmentos por encabezado y tamaño, con pruebas de bordes.
- [x] `docker-compose.yml` con PostgreSQL + pgvector y migración inicial.
- [x] Adaptador `VectorStore` (pgvector) con prueba de integración.
- [x] Puerto `EmbeddingProvider` y adaptador Ollama.
- [x] CLI `rag-ingest <carpeta>` idempotente por hash.

## M2 · Preguntas y citas (RF-02, RF-03)

Orden de abajo hacia arriba: `AskService` usa el LLM y el prompt, así que van antes.

- [x] Puerto `LLMProvider` y adaptador Ollama (con tope de tokens).
- [x] Prompt con fragmentos numerados y validación de citas.
- [x] `AskService` con umbral de relevancia y respuesta "no encontré información".
- [x] Endpoint `POST /ask` con validación (422) y pruebas de contrato.

## M3 · Proveedores e interfaz (RF-04, RF-05)

- [x] Adaptadores compatibles con OpenAI y con Anthropic; selección por `RAG_LLM_PROVIDER`.
- [x] Error explícito al iniciar si falta la clave del proveedor.
- [x] Página web de chat con indicador de carga y citas desplegables.

## M4 · Evaluación (RF-06)

- [x] 20+ preguntas de prueba en `eval/dataset.jsonl`.
- [x] Comando `rag-eval` con hit rate@k, MRR y fidelidad; reporte en `eval/reports/`.

## M4.1 · Mejoras guiadas por la evaluación (RF-02, RF-03, RF-06)

Cada mejora se mide con `rag-eval` antes y después; el reporte nuevo queda en `eval/reports/`.

- [x] Reconocer los rechazos que el modelo formula con sus propias palabras.
- [x] Búsqueda híbrida (vectorial + texto completo de PostgreSQL) para las paráfrasis.
- [x] Juez: los fragmentos citados conservan el número `[n]` que usa la respuesta.
- [x] Juez configurable aparte (`RAG_JUDGE_*`) y validado contra una revisión manual.

## M5 · Entrega

- [ ] `Dockerfile` de la API y `docker compose up` de punta a punta (RNF-05).
- [ ] Logs estructurados con id de petición (RNF-04).
- [ ] Despliegue público o video de demostración.
- [ ] README final con GIF, arquitectura y resultados de evaluación.
- [ ] Agregar el proyecto, con enlaces, a la sección Proyectos del CV.
