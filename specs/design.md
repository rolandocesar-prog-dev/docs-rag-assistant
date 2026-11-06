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
- **Ollama**: la app solo necesita `RAG_OLLAMA_URL` (por defecto `http://127.0.0.1:11434`). Hay
  dos formas de tenerlo:
  - **Instalado en el equipo** (recomendado si hay GPU): usa la GPU y es ≈ 8 veces más rápido en
    la medición de la sección de embeddings. Es la opción en el equipo de desarrollo (Windows,
    Radeon RX 6700 XT por Vulkan).
  - **En Docker**, para quien no tenga GPU o no quiera instalarlo: es opcional, con el *profile*
    `ollama` (`docker compose --profile ollama up -d`), y levanta `ollama/ollama:0.40.14` en
    `127.0.0.1:11434` con los modelos en un volumen. En Windows, Docker no usa GPUs AMD: corre en
    CPU.
  - En ambos casos los modelos se descargan una vez: `ollama pull bge-m3` (1,2 GB) y
    `ollama pull qwen3.5:4b` (3,3 GB); en Docker, anteponiendo `docker compose exec ollama`.

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

### Almacén vectorial (RF-01, RF-02)

- `EmbeddedChunk(chunk, embedding)` une un fragmento con su vector; `RetrievedChunk(chunk,
  document_title, source_path, score)` es un resultado de búsqueda, con lo necesario para citar
  (RF-03). `score` es la similitud de coseno (1 = idéntico).
- El puerto `VectorStore` define:
  - `content_hash(source_path) -> str | None`: hash guardado del documento, para no reingestarlo
    si no cambió.
  - `replace_document(document, chunks)`: borra la versión anterior del mismo `source_path` (sus
    fragmentos caen por `ON DELETE CASCADE`) e inserta la nueva, **en una sola transacción**: si
    algo falla, queda la versión anterior intacta.
  - `search(embedding, *, k, text=None) -> list[RetrievedChunk]`: los `k` fragmentos más
    relevantes. Sin `text`, por similitud de coseno; con `text`, **búsqueda híbrida** (abajo).
    `score` es siempre la similitud de coseno, así que el umbral de relevancia, que aplica
    `AskService` (RF-02), significa lo mismo en los dos modos.
- El adaptador `PgVectorStore` usa SQLAlchemy Core con SQL explícito y el operador `<=>`
  (distancia de coseno) de pgvector, que aprovecha el índice HNSW. Los vectores viajan como texto
  (`'[0.1, …]'::vector`), sin dependencias extra.
- **Búsqueda híbrida** (hallazgo de la evaluación: las paráfrasis, como «contraseña» para
  Secrets, quedan fuera de los 4 primeros solo con vectores):
  - La migración `0002` agrega a `chunks` la columna generada `content_tsv`
    (`to_tsvector('spanish', sección + contenido)`) con un índice GIN. Al ser generada, se
    calcula sola para los fragmentos ya guardados: no hace falta reingestar.
  - La pregunta se convierte en una consulta de texto completo en español que une sus palabras
    con **O** (cualquiera suma), no con Y, porque una pregunta casi nunca repite todas las
    palabras del documento. El analizador en español reduce las palabras a su raíz, así
    «contraseña» encuentra «contraseñas».
  - **Solo cuentan los términos raros**: los que aparecen en a lo sumo el 2 % de los fragmentos
    (como mínimo 1). Las palabras frecuentes («cómo» → `com`, «clúster», «forma») coinciden con la
    mitad del corpus y solo traen ruido. La frecuencia de cada término de la pregunta se cuenta
    en la misma consulta con el índice GIN (milisegundos; no hace falta una tabla aparte).
  - Se toman los 20 mejores por vector y los 5 mejores por texto, y se combinan con **Reciprocal
    Rank Fusion**: `1/(60 + posición)` sumado en las dos listas. Es el método estándar para unir
    rankings con puntuaciones no comparables. La lista de texto es corta porque su papel es
    rescatar lo que el vector no ve, no competir con él.
  - Si la pregunta no tiene términos raros, el resultado es el de la búsqueda vectorial.
  - `RAG_HYBRID_SEARCH` (por defecto `true`) permite comparar los dos modos con `rag-eval`.

### Embeddings (RF-01, RF-04)

- El puerto `EmbeddingProvider` define `embed(texts) -> list[tuple[float, ...]]`: un vector por
  texto, en el mismo orden.
- `EMBEDDING_DIMENSIONS = 1024` (dominio) es la dimensión del esquema. Si el proveedor no responde,
  no tiene el modelo, devuelve otra cantidad de vectores o vectores de otra dimensión, el
  adaptador lanza `EmbeddingError` con un mensaje explícito.
- El adaptador `OllamaEmbeddings` llama a `POST /api/embed` con `{"model", "input": [...]}`, por
  lotes (`RAG_EMBEDDING_BATCH_SIZE`, 32 por defecto), usando `httpx` con tiempo de espera.
  Configuración: `RAG_OLLAMA_URL` y `RAG_EMBEDDING_MODEL` (`bge-m3`).
- El adaptador compatible con OpenAI se agrega en M3, sobre el mismo puerto.
- Pruebas: unitarias con `httpx2.MockTransport` (sin red); integración contra Ollama real solo si
  `RAG_TEST_OLLAMA_URL` está definida. En CI no se ejecuta, porque el modelo pesa 1,2 GB.
- Cliente HTTP: `httpx2`, sucesor mantenido de `httpx` (que Starlette ya marca como obsoleto).
- Rendimiento medido (2026-11-05, Ollama en Docker solo con CPU, Ryzen 5 8600G): el corpus
  completo (862 fragmentos) tarda ≈ 5,4 min, unos 2,7 fragmentos/s; embeber una pregunta tarda
  ≈ 40 ms. La ingesta es puntual y solo reprocesa documentos cambiados, así que la CPU alcanza.
  Con Ollama instalado y GPU (Radeon RX 6700 XT por Vulkan): el corpus tarda **40 s** (≈ 21,5
  fragmentos/s, 8 veces más rápido) y una pregunta ≈ 20 ms. Los vectores de CPU y de GPU
  coinciden (similitud de coseno ≥ 0,999 en 20 fragmentos al azar): cambiar de uno a otro no
  obliga a reingestar.

### Modelo de lenguaje local (RF-02, RF-03)

Se compararon modelos de Ollama como generador, con 5 preguntas reales sobre el corpus (4 del
tema y 1 fuera de él), 4 fragmentos de contexto y Ollama en Docker solo con CPU (2026-11-05):

| Modelo          | Media  | Peor caso | Citas válidas | Rechaza fuera de tema | Bucles   |
| --------------- | ------ | --------- | ------------- | --------------------- | -------- |
| `qwen3.5:4b`    | 17,9 s | 21,8 s    | 4/4           | sí                    | 0        |
| `granite4.2:3b` | 22,2 s | 39,6 s    | 4/4           | sí                    | 1        |
| `gemma4:e4b`    | —      | —         | —             | —                     | no carga |

- **Elegido: `qwen3.5:4b`** (Apache 2.0, 3,3 GB): respuestas breves en buen español, citas `[n]`
  correctas y la frase exacta de «no encontré información». `granite4.2:3b` entró en bucle y no
  respetaba el formato; `gemma4:e4b` (6,6 GB) no cabe en la memoria de Docker (≈ 7,3 GB) y la
  agotó.
- Parámetros del adaptador: `think: false`, `temperature` 0,1, `num_ctx` 8 192 y **tope de
  `num_predict`** (400 tokens), para que un modelo en bucle nunca deje la respuesta colgada.
- Latencia en CPU: ≈ 75 % del tiempo es leer el contexto (≈ 100 tokens/s) y luego genera
  ≈ 15 tokens/s. Con 3 fragmentos en vez de 4 la media solo baja a 16,4 s: acortar el contexto
  no resuelve la lentitud en CPU.
- **Con GPU** (Ollama instalado en Windows, Radeon RX 6700 XT por Vulkan; ROCm no la soporta), las
  mismas 5 preguntas: media **2,2 s**, peor caso 3,7 s, primera palabra 1,5 s y ≈ 87 tokens/s, con
  la misma calidad (citas 4/4 y rechazo fuera de tema). Así RNF-01 (< 6 s) se cumple también en
  local; sin GPU, se cumple con un proveedor remoto (M3).
- En Docker conviene un solo modelo de lenguaje cargado a la vez, además de `bge-m3`.

**Puerto `LLMProvider`** (`ports/llm_provider.py`):

- `complete(system, user) -> Completion`: una instrucción de sistema y un mensaje de usuario, el
  mínimo común entre Ollama, las API compatibles con OpenAI y Anthropic (RF-04).
- `Completion(text, truncated)` (dominio): `truncated` indica que la respuesta llegó al tope de
  tokens; `AskService` lo trata como una respuesta no confiable.
- `LLMError` (dominio): el proveedor no responde, no tiene el modelo o devuelve algo inválido.
- Adaptador `OllamaChat`: `POST /api/chat` sin *streaming*, con los parámetros de arriba;
  configuración `RAG_LLM_MODEL` (`qwen3.5:4b`) y `RAG_LLM_MAX_TOKENS` (400). Comparte con
  `OllamaEmbeddings` el manejo de errores HTTP (`adapters/ollama_http.py`): sin respuesta, modelo
  no descargado (404, con el comando para descargarlo) u otro error.

**Proveedores de LLM intercambiables (RF-04)** — `RAG_LLM_PROVIDER` elige el adaptador al armar la
app (`adapters/llm/factory.py`); el resto del código solo conoce el puerto `LLMProvider`:

| Proveedor              | Adaptador              | Modelo por defecto | Tope por defecto | Cliente                 |
| ---------------------- | ---------------------- | ------------------ | ---------------- | ----------------------- |
| `ollama` (por defecto) | `OllamaChat`           | `qwen3.5:4b`       | 400              | `httpx2`                |
| `openai_compatible`    | `OpenAICompatibleChat` | — (obligatorio)    | 1 024            | `httpx2`                |
| `anthropic`            | `AnthropicChat`        | `claude-opus-5-5`  | 4 096            | SDK oficial `anthropic` |

- `RAG_LLM_MODEL` y `RAG_LLM_MAX_TOKENS` son opcionales: si faltan, se usa el valor por defecto
  del proveedor. `RAG_LLM_API_KEY` es la clave de los proveedores remotos y `RAG_LLM_BASE_URL` la
  dirección del servicio compatible con OpenAI (por ejemplo `https://api.openai.com/v1`).
- **Validación al arrancar (RF-04)**: `validate_llm_settings` revisa que el proveedor elegido
  tenga lo que necesita (clave no vacía; en `openai_compatible`, además URL y modelo) sin crear
  clientes ni conectarse. La API la ejecuta en su arranque (*lifespan* de FastAPI): si falta algo,
  uvicorn se detiene con un mensaje que nombra la variable. La base de datos, en cambio, se valida
  al usarla (503), para que `/health` responda aunque la base no esté configurada.
- **Compatible con OpenAI**: `POST {base}/chat/completions` con `Authorization: Bearer`, mensajes
  `system` y `user`, `max_tokens` y `temperature` 0,1. Sirve para OpenAI y para cualquier servicio
  con la misma API (vLLM, LM Studio, el propio Ollama en `/v1`, etc.). `finish_reason: "length"`
  marca la respuesta como cortada. Comparte el manejo de errores HTTP con los adaptadores de Ollama.
- **Anthropic**: SDK oficial (`client.beta.messages.create`), que ya usa `httpx2` y reintenta solo
  los errores transitorios. En `claude-opus-5-5` el razonamiento interno siempre está activo y
  cuenta dentro de `max_tokens`, por eso el tope es mayor (4 096) y el esfuerzo es `low`, el
  recomendado para respuestas breves de chat. En los modelos que lo admiten se envía
  `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`): si los filtros de seguridad
  rechazan por error una pregunta legítima, el propio servicio la reintenta con otro modelo.
  `stop_reason: "max_tokens"` marca la respuesta como cortada y `"refusal"` lanza `LLMError`.
  Los errores del SDK se traducen a `LLMError` con un mensaje claro: clave inválida, modelo
  inexistente, límite de uso, error del servicio o sin conexión.
- Pruebas: unitarias con transportes simulados de `httpx2` (el SDK de Anthropic acepta un cliente
  HTTP propio), sin red ni claves.

### Cargadores (RF-01)

Todos leen UTF-8 (con o sin BOM), unifican los fines de línea en `\n`, quitan espacios al final de
cada línea y reducen las líneas en blanco consecutivas a una.

| Formato  | Extensiones          | Título                                              | Partes         |
| -------- | -------------------- | --------------------------------------------------- | -------------- |
| Markdown | `.md`, `.markdown`   | `title` del bloque YAML, primer encabezado o nombre | Una            |
| Texto    | `.txt`               | Nombre del archivo                                  | Una            |
| PDF      | `.pdf` (con `pypdf`) | Título de los metadatos o nombre del archivo        | Una por página |

El Markdown conserva sus encabezados, que usa el divisor de fragmentos. Además, se quitan el bloque
YAML inicial (`---`), los comentarios HTML, las anclas de los encabezados (`## Título {#id}`, para
que no ensucien la sección citada) y las etiquetas de Hugo (`{{< … >}}`, `{{% … %}}`):

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

**Caso de uso de ingesta** (`application/ingest.py`, `IngestService`):

- Recibe un archivo o una carpeta (recorrida en orden alfabético y de forma recursiva; se ignoran
  archivos y carpetas ocultos, los que empiezan con `.`).
- `source_path` es la ruta **relativa a la carpeta padre** de lo ingestado, con `/`: ingestar
  `data/corpus/kubernetes/concepts` guarda `concepts/workloads/pods/pod.md`. Así la ruta no depende
  del equipo y la idempotencia funciona en cualquier máquina, siempre que se ingeste la misma
  carpeta.
- Cada archivo termina en un estado: `ingestado` (con su número de fragmentos), `sin cambios`
  (mismo hash), `vacío` (`EmptyDocumentError`) o `formato no soportado`. Los vacíos y los no
  soportados se informan y no detienen el resto; un `EmbeddingError` sí detiene la ingesta.
- Los archivos que desaparecen de la carpeta no se borran de la base (fuera del alcance de v1).

**CLI** (`rag-ingest <ruta> --license <licencia>`, con `argparse`):

- La licencia es obligatoria y se guarda en cada documento (ver `data/SOURCES.md`).
- Muestra una línea por archivo y un resumen con los totales y el tiempo.
- Códigos de salida: `0` si todo se procesó; `1` si hubo archivos con formato no soportado (RF-01),
  si falló el proveedor de embeddings o si la base no responde; `2` por argumentos inválidos o si
  falta `RAG_DATABASE_URL`.

**Pregunta (RF-02, RF-03):** validar la pregunta → embedding de la pregunta → búsqueda por
similitud de coseno (k = 4, el valor medido en la comparación de modelos) → descartar resultados
bajo el umbral → si no queda ninguno, responder "no encontré información" → construir el prompt
con los fragmentos numerados → pedir al LLM que responda solo con ese contexto y cite con `[n]` →
validar que las citas existan → devolver la respuesta y las fuentes.

**Prompt y citas** (`domain/prompting.py`, lógica pura):

- La instrucción de sistema es la de la comparación de modelos: responder en español, en 2 a 5
  frases, solo con los fragmentos numerados, citando cada afirmación con `[n]`, y si no está la
  respuesta, decir exactamente «No encontré información sobre eso en la documentación.».
- El mensaje de usuario numera los fragmentos desde 1: `[1] Documento: Pods`, en otra línea
  `Sección: …` (si la hay) y luego el texto; termina con `Pregunta: …`.
- `Answer(text, citations, found)` y `Citation(id, document, section, page, source_path, excerpt)`
  son entidades del dominio. El extracto son los primeros ≈ 200 caracteres del fragmento, sin sus
  encabezados ni las URL de los enlaces Markdown (queda su texto), cortado en un límite de palabra.
- Validación de la respuesta del modelo:
  - Se reconocen `[2]`, `[1][3]` y `[1, 3]`. Los números que no corresponden a un fragmento se
    quitan del texto (y la marca entera, si no queda ninguno válido).
  - Las fuentes son solo los fragmentos citados, en orden y sin repetir, con su número original
    para que coincidan con las marcas del texto.
  - Una respuesta **sin ninguna cita válida** no está fundamentada (RF-03): se devuelve «no
    encontré información» con `found = false`, en vez de arriesgar una respuesta inventada (RF-02).
    Esto cubre también el caso en que el modelo dice que no encontró la respuesta.
  - Si hay citas válidas y además el modelo agregó la frase de «no encontré información», esa
    frase se quita.
  - **Rechazo con otras palabras** (hallazgo de la primera evaluación): si la **primera frase**
    dice que la documentación, los fragmentos o el contexto no contienen, no mencionan o no
    explican lo preguntado (p. ej. «La documentación proporcionada no contiene instrucciones
    sobre…»), la respuesta es un rechazo aunque cite fragmentos, y se devuelve «no encontré
    información». Solo se mira la primera frase para no descartar respuestas útiles que aclaran
    un límite más adelante («…; la documentación no menciona X»).

**Caso de uso de pregunta** (`application/ask.py`, `AskService(embedder, store, llm)`):

1. Valida la pregunta: sin espacios sobrantes, no vacía y de hasta 1 000 caracteres; si no,
   `InvalidQuestionError` (dominio), que la API traduce a HTTP 422 (RF-02).
2. Genera su embedding y busca los `k` fragmentos más similares (`RAG_RETRIEVAL_K`, 4).
3. Descarta los que no alcanzan el umbral (`RAG_RELEVANCE_THRESHOLD`, 0,45: con el corpus, las
   preguntas fuera de tema puntúan ≈ 0,3 y las del tema ≈ 0,6–0,7). Si no queda ninguno, responde
   «no encontré información» **sin llamar al LLM** (RF-02), lo que además ahorra tiempo.
4. Arma el prompt solo con los fragmentos relevantes, llama al LLM y valida las citas.
5. Si la respuesta llegó al tope de tokens (`truncated`), lanza `LLMError`: suele ser un bucle y
   no conviene mostrarla ni disfrazarla de «no encontré información», porque la información sí
   podría estar.

Comprobación del umbral con 10 preguntas reales (2026-11-05, GPU): las 6 del tema puntúan entre
0,56 y 0,72 y se responden con citas en 2–3 s; las 3 fuera de tema («hola», un mundial de fútbol,
la capital de Francia) puntúan 0,32–0,43 y se rechazan sin llamar al LLM. «¿Cómo instalo
Kubernetes en Windows?» (0,65) pasa el umbral, pero el LLM responde que no está en los fragmentos:
el umbral no tiene que ser perfecto, porque el prompt es una segunda barrera. Queda un fallo de
recuperación conocido: «guardar una contraseña» no trae la página de Secrets (se medirá en RF-06).

## 5. API

| Método | Ruta      | Descripción                               |
| ------ | --------- | ----------------------------------------- |
| GET    | `/health` | Estado y versión (ya implementado).       |
| POST   | `/ask`    | `{ "question": str }` → respuesta + citas |
| POST   | `/ingest` | Sube un archivo para ingesta.             |
| GET    | `/`       | Página web de chat.                       |

Respuesta de `/ask` (`200`, también cuando no encuentra información: `found` lo indica):

```json
{
  "answer": "Un Pod agrupa contenedores [1].",
  "citations": [
    {
      "id": 1,
      "document": "Pods",
      "section": "¿Qué es un Pod?",
      "page": null,
      "source_path": "concepts/workloads/pods/pod.md",
      "excerpt": "Un Pod es un grupo de uno o más contenedores…"
    }
  ],
  "found": true
}
```

Errores de `/ask`, siempre con `{"detail": …}`:

| Código | Cuándo                                                                                    |
| ------ | ----------------------------------------------------------------------------------------- |
| 422    | Falta `question`, no es texto, está vacía o solo con espacios, o supera 1 000 caracteres. |
| 502    | El proveedor de embeddings o el LLM falló o respondió algo inválido (incluye el tope).    |
| 503    | La base de datos no responde o falta `RAG_DATABASE_URL`.                                  |

El endpoint es síncrono (FastAPI lo ejecuta en su pool de hilos), porque los adaptadores lo son.
`AskService` se arma una sola vez (`api/dependencies.py`) y las pruebas lo reemplazan con dobles
mediante `dependency_overrides`. La CLI y la API crean la conexión a la base con la misma función
(`adapters/database.py`), con un tiempo de espera de 10 s.

### Interfaz web de chat (RF-05)

- `GET /` devuelve `web/index.html`; `web/app.js` y `web/styles.css` se sirven en `/static`. Sin
  framework, sin dependencias externas ni CDN: todo sale del mismo servidor, así que no hace falta
  CORS.
- La pregunta va en un cuadro de texto con contador (máximo 1 000 caracteres, como la API); Enter
  envía y Mayús+Enter agrega una línea.
- Mientras espera la respuesta, la conversación muestra un indicador animado («Buscando en la
  documentación…») y el botón queda desactivado para no enviar dos veces (RF-05).
- La respuesta se muestra con sus marcas `[n]` convertidas en botones; debajo, cada cita es una
  tarjeta desplegable (`<details>`) con documento, sección, página y extracto. Pulsar `[n]` abre y
  resalta su tarjeta (RF-03).
- `found: false` se muestra como aviso neutro; los errores (422, 502, 503) muestran el `detail`
  de la API.
- **Seguridad**: el texto de la API nunca se inserta como HTML (`textContent` y nodos creados con
  `document.createElement`), así una respuesta del modelo no puede inyectar código. Una prueba
  verifica que `app.js` no use `innerHTML`.
- Accesible: etiquetas en los controles, `aria-live` en la conversación, foco visible y respeta
  el modo claro u oscuro del sistema.

## 6. Decisiones

| Decisión                    | Elección                     | Motivo                                                                    |
| --------------------------- | ---------------------------- | ------------------------------------------------------------------------- |
| Base vectorial              | PostgreSQL + pgvector        | Un solo servicio para datos y vectores, SQL conocido, fácil en Docker.    |
| Framework web               | FastAPI                      | Tipado con Pydantic, OpenAPI automático, asíncrono.                       |
| Orquestación RAG            | Código propio, sin LangChain | Muestra comprensión del flujo y evita una dependencia pesada.             |
| Proveedor de modelo inicial | Ollama (local), `qwen3.5:4b` | Desarrollo sin costo; los demás proveedores entran por el mismo puerto.   |
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

## 8. Evaluación (RF-06)

**Conjunto de preguntas** (`eval/dataset.jsonl`, una pregunta por línea, versionado):

```json
{"id": "q01", "category": "definicion", "question": "¿Qué es un Pod?",
 "expected": [{"source_path": "concepts/workloads/pods/pod.md", "section": "¿Qué és un Pod?"}],
 "reference": "Un Pod es un grupo de uno o más contenedores…"}
```

- `category`: `definicion`, `procedimiento`, `comparacion`, `parafrasis` (la pregunta usa palabras
  distintas a las del documento, como «contraseña» para Secrets) o `fuera_de_tema`.
- `expected`: los fragmentos que responden la pregunta, por documento y sección. La sección se
  escribe como la ruta de encabezados que guarda el divisor (`null` es el texto anterior al primer
  encabezado). En `fuera_de_tema` la lista está vacía y `reference` es `null`: la respuesta
  correcta es «no encontré información».
- `reference`: respuesta de referencia redactada a partir del texto del corpus, sin agregar nada
  que no diga.
- Al menos 20 preguntas del tema y algunas fuera de tema. Una prueba (sin base ni modelo) verifica
  que el archivo sea válido y que cada documento y sección esperados existan en el corpus, así un
  cambio del corpus o del divisor no deja el conjunto desactualizado sin avisar.

**Métricas** (`domain/evaluation.py`, lógica pura):

- **Recuperación**, sobre las preguntas del tema, buscando los 10 fragmentos más similares sin
  umbral: posición del primer fragmento relevante, *hit rate@1, @4 y @10* (¿aparece entre los
  primeros k?) y **MRR@10** (media de 1/posición; 0 si no aparece). Se calcula a nivel de
  **documento** (mismo `source_path`) y de **sección** (además, la misma sección), en total y
  por categoría. @4 es el caso real: `AskService` usa k = 4.
- **Respuestas**, pasando cada pregunta por `AskService` (umbral y prompt reales):
  - tasa de respuesta en las preguntas del tema y **tasa de rechazo** en las de fuera de tema;
  - **cita la fuente esperada**: la respuesta cita al menos un fragmento de un documento esperado;
  - **fidelidad** y **corrección**, con un **LLM como juez**: recibe la pregunta, los fragmentos
    citados, la respuesta y la referencia, y responde solo
    `{"fiel": true|false, "correcta": true|false}`. *Fiel* = todo lo que afirma la respuesta está
    en los fragmentos citados; *correcta* = coincide en lo esencial con la referencia. Un veredicto
    ilegible cuenta como «sin veredicto», no como acierto. Cada fragmento citado llega al juez con
    **el mismo número `[n]` que usa la respuesta**: si la respuesta cita `[2]` y `[4]`, el juez
    ve `[2]` y `[4]`, no `[1]` y `[2]`;
  - **latencia** p50 y p95 de cada respuesta completa (RNF-01).
- Un error del modelo en una pregunta (por ejemplo, el tope de tokens) queda registrado en esa
  pregunta y no detiene la evaluación.

**CLI** `rag-eval [--k 10] [--sin-juez]`: usa la configuración de `.env` (la base ingestada y el
proveedor de LLM elegido) y escribe en `eval/reports/` un JSON completo y un Markdown legible con
el mismo nombre (fecha, hora y modelo). Los reportes se versionan: el último se cita en el
README.

**Juez configurable**: `RAG_JUDGE_PROVIDER`, `RAG_JUDGE_MODEL`, `RAG_JUDGE_MAX_TOKENS`,
`RAG_JUDGE_API_KEY` y `RAG_JUDGE_BASE_URL` funcionan igual que sus pares `RAG_LLM_*` (mismos
proveedores y valores por defecto). Sin `RAG_JUDGE_PROVIDER`, el juez es el mismo modelo que
responde. La fábrica de LLM arma los dos a partir de una configuración común (`LLMConfig`), y los
mensajes de error nombran la variable del grupo que falta. El reporte muestra los dos modelos.
`rag-eval` valida la configuración del juez antes de empezar, salvo con `--sin-juez`.

**Primera evaluación** (2026-11-05, `ollama/qwen3.5:4b` con GPU, reporte en `eval/reports/`):
recuperación por documento hit@4 95 %, hit@10 100 %, MRR 0,86 (por sección, hit@4 65 %);
responde 19/20 preguntas del tema y rechaza 3/4 fuera de tema; latencia p50 2,1 s y p95 3,3 s
(RNF-01 se cumple en local con GPU). Hallazgos, que guían las mejoras siguientes:

- **El juez con el mismo modelo de 4B no es fiable**: marcó como infieles o incorrectas respuestas
  que sí lo eran (por ejemplo, `kubectl rollout undo --to-revision` para revertir un Deployment).
  Fidelidad (68 %) y corrección (60 %) son cotas inferiores ruidosas; para medirlas en serio, el
  juez debe ser un modelo más capaz (se puede elegir otro proveedor por configuración).
- **Rechazos con otras palabras**: en «instalar Python en Windows» y en «contraseña» el modelo
  dijo que la documentación no lo cubre, pero con sus palabras y citando fragmentos, así que cuenta
  como respuesta. Hay que reforzar la frase exacta en el prompt o reconocer esos rechazos.
- **Paráfrasis**: «contraseña» trae la página de Secrets en el puesto 6 (fuera de k = 4), y «se
  cae una máquina» recupera bien el documento pero el modelo no responde. Es el caso para probar
  búsqueda híbrida (vectorial + palabras clave) o reordenamiento, y medirlo con este mismo
  conjunto.

**Mejora 1 — rechazos parafraseados** (reporte `2026-11-05-2142`): rechaza 4/4 preguntas fuera de
tema (antes 3/4) y responde 18/20 del tema (antes 19/20: «contraseña» ahora dice que no encontró
información, porque la búsqueda no le entregó la página de Secrets). La recuperación no cambia,
como se esperaba. Revisé el detector sobre las 20 respuestas de la primera evaluación: solo marcó
los dos rechazos reales.

**Mejora 2 — búsqueda híbrida** (reporte `2026-11-05-2152`):

- **Primer intento, peor**: con todas las palabras de la pregunta y 20 candidatos por lista, la
  recuperación empeoró (documento hit@4 85 %, MRR 0,69; paráfrasis hit@4 50 %). La consulta de
  «contraseña» coincidía con 436 de 862 fragmentos: `com` (de «cómo») está en 151, `clust` en 210
  y `form` en 170, frente a 7 de `contraseñ`.
- **Ajuste** (solo recuperación, sin LLM, con las 20 preguntas del tema): probé un umbral de
  frecuencia (100 %, 10 %, 5 %, 2 %), el peso de la lista de texto (1 o 0,5) y sus candidatos
  (20 o 5). Con 2 %, peso 1 y 5 candidatos todas las métricas igualan o superan a la búsqueda
  vectorial; con umbrales más altos siempre hubo alguna peor.
- **Resultado** (contra la mejora 1): documento hit@4 **100 %** (antes 95 %), MRR **0,88**
  (0,86), hit@1 igual (80 %); sección hit@4 **70 %** (65 %). «Contraseña» trae Secrets en el
  puesto 2 (antes 6) y ahora se responde citándolo. Responde 19/20 del tema, rechaza 4/4 fuera de
  tema, cita la fuente esperada 17/20 (antes 15/20); latencia p50 2,0 s y p95 3,0 s (sin costo
  visible).
- **Cuidado**: los tres valores se eligieron mirando estas mismas 20 preguntas, así que la mejora
  puede estar algo sobreestimada; con un conjunto más grande habría que separar preguntas de
  ajuste y de prueba. Queda pendiente «se cae una máquina» (q18): recupera bien el documento, pero
  el modelo responde que no encontró información.

**Mejora 3a — numeración de los fragmentos del juez** (reporte `2026-11-05-2204`): el juez
recibía los fragmentos citados renumerados desde `[1]`, mientras la respuesta conservaba sus
números (por ejemplo `[2]` y `[4]`), así que veía citas a fragmentos inexistentes. Con la
numeración corregida y el mismo juez de 4B, la fidelidad pasó de 12/19 (63 %) a **18/19 (95 %)**
y la corrección de 13/20 a 14/20; la recuperación y las respuestas no cambian (hit@4 100 %,
19/20, 4/4). La fidelidad baja que se atribuía solo al tamaño del juez venía en buena parte de este
defecto.

**Mejora 3b — juez de 9B** (`RAG_JUDGE_PROVIDER=ollama`, `RAG_JUDGE_MODEL=qwen3.5:9b`, 6,6 GB;
cabe en la GPU junto al modelo que responde y `bge-m3`):

- **Validación**: generé las 19 respuestas una sola vez, las pasé por los dos jueces y revisé
  cada veredicto a mano contra los fragmentos citados y la referencia (38 veredictos):

  | Juez         | Coincide en fidelidad | Coincide en corrección | Veredictos errados |
  | ------------ | --------------------- | ---------------------- | ------------------ |
  | `qwen3.5:4b` | 16/19                 | 15/19                  | 7 de 38            |
  | `qwen3.5:9b` | 17/19                 | 19/19                  | 2 de 38            |

  El de 4B rechazaba respuestas correctas (namespace, TTL de Jobs, `rollout undo`, Secrets). Los
  dos errores del de 9B son por indulgencia en fidelidad: aprobó afirmaciones que el fragmento no
  dice («hay que especificar `hostIP`», «los datos de un volumen `secret` pueden perderse al
  reiniciar»). Su fidelidad es, entonces, una cota superior.
- **Reporte** `2026-11-05-2215` (responde `qwen3.5:4b`, juzga `qwen3.5:9b`): fidelidad 17/19
  (89 %) y corrección 15/20 (75 %); recuperación y respuestas iguales a la mejora 2 (hit@4 100 %,
  19/20, 4/4). Las respuestas incorrectas son de procedimiento y paráfrasis: hostPort en lugar de
  NodePort, «varias copias» respondida con zonas de disponibilidad (en realidad un rechazo en la
  segunda frase, que el detector no ve porque solo mira la primera) y volúmenes de StorageOS para
  «datos que no se pierdan».
- Sin `RAG_JUDGE_*`, el juez sigue siendo el mismo modelo que responde: así un proveedor remoto
  más capaz se puede usar como juez sin tocar código.
