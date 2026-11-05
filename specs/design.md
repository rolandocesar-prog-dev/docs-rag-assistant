# Diseño — Asistente RAG sobre documentación pública

Responde al **cómo** de [`requirements.md`](./requirements.md). Las decisiones importantes se
registran en la sección 6 con su justificación.

## 1. Arquitectura

Arquitectura hexagonal (puertos y adaptadores): el dominio no conoce FastAPI, la base de datos ni
el proveedor del modelo. Eso permite cambiar de proveedor (RF-04) y probar la lógica sin red.

```
            ┌──────────────── Adaptadores de entrada ────────────────┐
            │  API REST (FastAPI)     CLI de ingesta     Web de chat  │
            └───────────────┬─────────────────────────────────────────┘
                            │
                ┌───────────▼───────────┐
                │      Aplicación       │  IngestService · AskService · EvaluationService
                └───────────┬───────────┘
                            │ usa puertos (interfaces)
                ┌───────────▼───────────┐
                │        Dominio        │  Document · Chunk · Answer · Citation
                └───────────┬───────────┘
                            │
            ┌───────────────▼─────────────── Adaptadores de salida ───────────┐
            │ DocumentLoader (md/txt/pdf)  EmbeddingProvider  LLMProvider       │
            │ VectorStore (PostgreSQL + pgvector)                               │
            └───────────────────────────────────────────────────────────────────┘
```

## 2. Estructura del código

```
src/rag_assistant/
├── domain/          # entidades y reglas puras (sin dependencias externas)
├── application/     # casos de uso: ingest, ask, evaluate
├── ports/           # interfaces (typing.Protocol)
├── adapters/
│   ├── loaders/     # markdown, txt, pdf
│   ├── llm/         # ollama, openai_compatible, anthropic
│   ├── embeddings/
│   └── vectorstore/ # pgvector
├── api/             # routers FastAPI (health, ask, ingest)
├── web/             # página de chat (HTML + JS sin framework)
├── config.py
└── main.py
```

## 3. Modelo de datos

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id           UUID PRIMARY KEY,
    source_path  TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    license      TEXT NOT NULL,
    content_hash TEXT NOT NULL,       -- RF-01: evita reingestas duplicadas
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE chunks (
    id          UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    position    INT  NOT NULL,
    section     TEXT,
    page        INT,
    content     TEXT NOT NULL,
    embedding   vector(768) NOT NULL  -- la dimensión depende del modelo de embeddings
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

### Entidades del dominio

Son inmutables (`dataclass(frozen=True)`) y se validan al crearse, de modo que un objeto inválido
no puede existir.

- `Document`: `source_path`, `title`, `license` y `parts`, una tupla de `DocumentPart(text, page)`.
  Los formatos sin páginas (Markdown, TXT) producen una sola parte con `page=None`; el PDF, una
  por página. `content_hash` es el SHA-256 de las partes (texto y página), sin título ni licencia:
  si el contenido no cambió, no se reingesta (RF-01).
- `Chunk`: `document_id`, `position`, `content`, `section` y `page`. El embedding se asocia al
  guardarlo en el `VectorStore`; no forma parte de la entidad.
- `UnsupportedFormatError`: la ingesta la lanza si ningún cargador acepta el archivo; la API la
  traduce a HTTP 415 y la CLI a un código de salida distinto de 0 (RF-01).

El puerto `DocumentLoader` (en `ports/`) define `supports(path) -> bool` y
`load(path, *, license) -> Document`. Cada formato es un adaptador en `adapters/loaders/`.

## 4. Flujos

**Ingesta (RF-01):** cargar archivo → normalizar texto → calcular hash → si no cambió, terminar →
dividir por encabezados y luego por tamaño (≈ 800 tokens, solapamiento de 100) → generar
embeddings por lotes → reemplazar los fragmentos del documento en una sola transacción.

**Pregunta (RF-02, RF-03):** validar la pregunta → embedding de la pregunta → búsqueda por
similitud de coseno (k = 5) → descartar resultados bajo el umbral → si no queda ninguno, responder
"no encontré información" → construir el prompt con los fragmentos numerados → pedir al LLM que
responda solo con ese contexto y cite con `[n]` → validar que las citas existan → devolver la
respuesta y las fuentes.

## 5. API

| Método | Ruta      | Descripción                               |
| ------ | --------- | ----------------------------------------- |
| GET    | `/health` | Estado y versión (ya implementado).       |
| POST   | `/ask`    | `{ "question": str }` → respuesta + citas |
| POST   | `/ingest` | Sube un archivo para ingesta.             |
| GET    | `/`       | Página web de chat.                       |

Respuesta de `/ask`:

```json
{
  "answer": "Texto con citas [1][2].",
  "citations": [{ "id": 1, "document": "guia.md", "section": "Instalación", "excerpt": "..." }],
  "found": true
}
```

## 6. Decisiones

| Decisión                    | Elección                     | Motivo                                                                  |
| --------------------------- | ---------------------------- | ----------------------------------------------------------------------- |
| Base vectorial              | PostgreSQL + pgvector        | Un solo servicio para datos y vectores, SQL conocido, fácil en Docker.  |
| Framework web               | FastAPI                      | Tipado con Pydantic, OpenAPI automático, asíncrono.                     |
| Orquestación RAG            | Código propio, sin LangChain | Muestra comprensión del flujo y evita una dependencia pesada.           |
| Proveedor de modelo inicial | Ollama (local)               | Desarrollo sin costo; los demás proveedores entran por el mismo puerto. |
| Interfaz                    | HTML + JS sin framework      | El foco del proyecto es el backend de IA.                               |
| Corpus de demostración      | Kubernetes, «Conceptos» (ES) | Traducción humana revisada, CC BY 4.0 y temas que entiende cualquiera.  |

### Corpus de demostración

- Sección «Conceptos» de la documentación de Kubernetes en español
  (`kubernetes/website`, `content/es/docs/concepts`), con licencia CC BY 4.0.
- Se versiona una copia **sin modificar** en `data/corpus/kubernetes/`, fijada a un commit del
  repositorio de origen. Así la evaluación (RF-06) es reproducible y `docker compose up` no depende
  de descargas. El commit, la licencia y la atribución se registran en `data/SOURCES.md`.
- La limpieza (metadatos YAML, etiquetas de Hugo, comentarios HTML) la hace el cargador de
  Markdown, no se aplica a los archivos.
- Como el corpus está en español, el modelo de embeddings debe ser multilingüe; se elige en la
  tarea del adaptador de embeddings y se registra en esta tabla.

## 7. Estrategia de pruebas

- **Unitarias:** dominio y servicios con dobles de prueba de los puertos (sin red ni base de datos).
- **Integración:** adaptador pgvector contra PostgreSQL en Docker (servicio de CI).
- **Contrato de API:** `TestClient` de FastAPI para cada endpoint.
- **Calidad del RAG:** RF-06, con un reporte versionado en `eval/reports/`.
