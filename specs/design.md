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
    position    INT  NOT NULL CHECK (position >= 0),
    section     TEXT,
    page        INT  CHECK (page >= 1),
    content     TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,  -- dimensión de bge-m3 (ver sección 6)
    UNIQUE (document_id, position)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

El esquema se crea y versiona con **Alembic** (`migrations/`), con SQL explícito en cada migración.
La URL de conexión llega por `RAG_DATABASE_URL` (RNF-02).

### Base de datos local y pruebas de integración

- `docker compose up -d db` levanta `pgvector/pgvector:0.8.7-pg18` con las bases `rag` y
  `rag_test`, publicada solo en `127.0.0.1:5442` (5432 y 5433 suelen estar ocupados por otros
  proyectos). La contraseña sale de `RAG_DB_PASSWORD` en `.env`.
- `alembic upgrade head` aplica las migraciones sobre `RAG_DATABASE_URL`.
- Las pruebas de integración usan `RAG_TEST_DATABASE_URL`. Si no está definida, se omiten y Pytest
  lo muestra; si está definida y la base no responde, fallan. En CI, un servicio de PostgreSQL con
  pgvector la define, así que siempre se ejecutan.

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

Si tras la limpieza un archivo no tiene texto (por ejemplo, un `_index.md` con solo metadatos o un
PDF escaneado), `Document` lanza `EmptyDocumentError` y la ingesta lo omite informándolo.

### Cargadores (RF-01)

Todos leen UTF-8 (con o sin BOM), unifican los fines de línea en `\n`, quitan espacios al final de
cada línea y reducen las líneas en blanco consecutivas a una.

| Formato  | Extensiones          | Título                                              | Partes         |
| -------- | -------------------- | --------------------------------------------------- | -------------- |
| Markdown | `.md`, `.markdown`   | `title` del bloque YAML, primer encabezado o nombre | Una            |
| Texto    | `.txt`               | Nombre del archivo                                  | Una            |
| PDF      | `.pdf` (con `pypdf`) | Título de los metadatos o nombre del archivo        | Una por página |

El Markdown conserva sus encabezados, que usa el divisor de fragmentos. Además, se quitan el bloque
YAML inicial (`---`), los comentarios HTML y las etiquetas de Hugo (`{{< … >}}`, `{{% … %}}`):

| Etiqueta                                         | Se reemplaza por                                           |
| ------------------------------------------------ | ---------------------------------------------------------- |
| `glossary_tooltip`, `link`                       | Su `text` (o `term_id`), para no cortar la frase.          |
| `figure`, `table`, `alert`, `tab`                | Su `caption`, `title` o `name`.                            |
| `feature-state`                                  | «Estado de la función: Kubernetes vX [estado]».            |
| `heading "whatsnext"`                            | «¿Qué sigue?».                                             |
| `comment` … `/comment`                           | Nada: se elimina con su contenido.                         |
| Cualquier otra (`note`, `param`, `code_sample`…) | Nada: se quita la etiqueta y se conserva el texto interno. |

Las páginas de PDF sin texto se omiten; el número de página de las demás se conserva.

## 4. Flujos

**Ingesta (RF-01):** cargar archivo → normalizar texto → calcular hash → si no cambió, terminar →
dividir por encabezados y luego por tamaño (≈ 800 tokens, solapamiento de 100) → generar
embeddings por lotes → reemplazar los fragmentos del documento en una sola transacción.

**División en fragmentos** (`domain/chunking.py`, lógica pura):

1. Cada parte del documento se divide por separado, así un fragmento nunca mezcla páginas.
2. Los encabezados de Markdown abren secciones; los `#` dentro de bloques de código no cuentan.
   La sección del fragmento es la ruta de encabezados (`Service > Tipos de Service > ClusterIP`) y
   es `None` para el texto anterior al primer encabezado. Una sección sin cuerpo no genera
   fragmento, pero su título sigue en la ruta de sus subsecciones.
3. Una sección que supera el tamaño máximo se reparte por párrafos (un bloque de código cuenta
   como un párrafo). Un párrafo demasiado grande se corta por líneas, luego por palabras y, como
   último recurso, por caracteres.
4. Cada fragmento siguiente de la misma sección empieza con el final del anterior (solapamiento),
   cortado en un límite de palabra. Ningún fragmento supera el máximo.

El tamaño se mide en caracteres para no depender del tokenizador: por defecto, 3 200 de máximo y
400 de solapamiento (≈ 800 y 100 tokens, con unos 4 caracteres por token). Ambos son parámetros y
quedan muy por debajo del límite de `bge-m3` (8 192 tokens).

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

| Decisión                    | Elección                     | Motivo                                                                    |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| Base vectorial              | PostgreSQL + pgvector        | Un solo servicio para datos y vectores, SQL conocido, fácil en Docker.    |
| Framework web               | FastAPI                      | Tipado con Pydantic, OpenAPI automático, asíncrono.                       |
| Orquestación RAG            | Código propio, sin LangChain | Muestra comprensión del flujo y evita una dependencia pesada.             |
| Proveedor de modelo inicial | Ollama (local)               | Desarrollo sin costo; los demás proveedores entran por el mismo puerto.   |
| Interfaz                    | HTML + JS sin framework      | El foco del proyecto es el backend de IA.                                 |
| Corpus de demostración      | Kubernetes, «Conceptos» (ES) | Traducción humana revisada, CC BY 4.0 y temas que entiende cualquiera.    |
| Modelo de embeddings        | `bge-m3` (Ollama), 1 024 dim | Multilingüe (100+ idiomas), entradas de hasta 8 192 tokens, licencia MIT. |
| Migraciones                 | Alembic con SQL explícito    | Estándar en Python, versionadas y reversibles; el SQL queda a la vista.   |

### Corpus de demostración

- Sección «Conceptos» de la documentación de Kubernetes en español
  (`kubernetes/website`, `content/es/docs/concepts`), con licencia CC BY 4.0.
- Se versiona una copia **sin modificar** en `data/corpus/kubernetes/`, fijada a un commit del
  repositorio de origen. Así la evaluación (RF-06) es reproducible y `docker compose up` no depende
  de descargas. El commit, la licencia y la atribución se registran en `data/SOURCES.md`.
- La limpieza (metadatos YAML, etiquetas de Hugo, comentarios HTML) la hace el cargador de
  Markdown, no se aplica a los archivos.
- Como el corpus está en español, el modelo de embeddings debe ser multilingüe: se eligió
  `bge-m3` (ver la tabla de decisiones).

## 7. Estrategia de pruebas

- **Unitarias:** dominio y servicios con dobles de prueba de los puertos (sin red ni base de datos).
- **Integración:** adaptador pgvector contra PostgreSQL en Docker (servicio de CI).
- **Contrato de API:** `TestClient` de FastAPI para cada endpoint.
- **Calidad del RAG:** RF-06, con un reporte versionado en `eval/reports/`.
