# Requisitos — Asistente RAG sobre documentación pública

> Spec Driven Development: este archivo define **qué** debe hacer el sistema. El **cómo** está en
> [`design.md`](./design.md) y el plan de trabajo en [`tasks.md`](./tasks.md). Ningún cambio de
> código se hace sin que esté respaldado por un requisito de este documento.

## 1. Objetivo

Permitir que cualquier persona haga preguntas en lenguaje natural sobre un conjunto de documentos
públicos y reciba respuestas **fundamentadas únicamente en esos documentos**, con citas que indiquen
de qué documento y sección sale cada afirmación.

## 2. Alcance

**Incluye:** ingesta de documentos, búsqueda semántica, generación de respuestas con citas,
interfaz web mínima de chat, evaluación automática de calidad y despliegue con Docker.

**Fuera de alcance (v1):** cuentas de usuario y permisos, OCR de documentos escaneados, historial
de conversaciones persistente, entrenamiento o ajuste fino de modelos.

## 3. Corpus de datos

- Solo documentos **públicos y con licencia que permita su reutilización** (por ejemplo,
  documentación de proyectos open source con licencia MIT, Apache 2.0 o CC BY).
- La licencia de cada documento se registra en `data/SOURCES.md`.
- No se reutilizan código, datos ni documentos de proyectos de clientes ni del proyecto de grado.

## 4. Requisitos funcionales

Los criterios de aceptación usan el formato EARS (_CUANDO / SI / MIENTRAS … el sistema DEBE …_).

### RF-01 · Ingesta de documentos

**Historia:** Como administrador, quiero cargar documentos para que el asistente pueda responder
sobre ellos.

- CUANDO se ingesta un archivo Markdown, TXT o PDF con texto, el sistema DEBE dividirlo en
  fragmentos, generar un embedding por fragmento y guardarlo con sus metadatos (documento, título
  de sección, página si aplica).
- SI el formato no está soportado, el sistema DEBE rechazarlo con un error claro (HTTP 415 en la
  API, código de salida distinto de 0 en la CLI).
- CUANDO se ingesta de nuevo un documento sin cambios (mismo hash de contenido), el sistema NO DEBE
  duplicar sus fragmentos.
- CUANDO se ingesta una nueva versión de un documento, el sistema DEBE reemplazar sus fragmentos
  anteriores.

### RF-02 · Preguntar

**Historia:** Como usuario, quiero hacer una pregunta y recibir una respuesta breve y correcta.

- CUANDO se envía una pregunta a `POST /ask`, el sistema DEBE recuperar los fragmentos más
  relevantes y generar una respuesta usando solo ese contexto.
- SI ningún fragmento supera el umbral de relevancia configurado, el sistema DEBE responder que no
  encontró información en los documentos, **sin inventar una respuesta**.
- SI la pregunta está vacía o supera 1 000 caracteres, el sistema DEBE responder HTTP 422.

### RF-03 · Citas a la fuente

- Cada respuesta DEBE incluir la lista de fuentes usadas: documento, sección y un extracto.
- Cada afirmación de la respuesta DEBE poder relacionarse con al menos una cita (marcadores `[1]`,
  `[2]`, …).

### RF-04 · Proveedor de modelos intercambiable

- El sistema DEBE permitir cambiar el proveedor de LLM por configuración, sin cambiar código:
  modelo local (Ollama), API compatible con OpenAI y API de Anthropic.
- El sistema DEBE permitir cambiar el proveedor de embeddings por configuración, sin cambiar
  código: modelo local (Ollama) y API compatible con OpenAI. (Anthropic no ofrece un modelo de
  embeddings.)
- SI el proveedor de embeddings devuelve vectores de una dimensión distinta a la del esquema, el
  sistema DEBE detener la ingesta con un error explícito.
- SI falta la clave de API del proveedor configurado, el sistema DEBE fallar al iniciar con un
  mensaje explícito.

### RF-05 · Interfaz web

- El sistema DEBE ofrecer una página web de chat que muestre la respuesta y, debajo, las citas
  como enlaces o tarjetas desplegables.
- MIENTRAS se genera la respuesta, la interfaz DEBE mostrar un indicador de carga.

### RF-06 · Evaluación de calidad

- El repositorio DEBE incluir un conjunto de al menos 20 preguntas de prueba con la sección
  esperada y la respuesta de referencia.
- Un comando DEBE calcular métricas de recuperación (_hit rate@k_, MRR) y de fidelidad de la
  respuesta, y guardar el resultado en un reporte.

## 5. Requisitos no funcionales

| ID     | Requisito                                                                      |
| ------ | ------------------------------------------------------------------------------ |
| RNF-01 | Latencia p95 de `POST /ask` menor a 6 s con un modelo remoto.                  |
| RNF-02 | Secretos solo por variables de entorno; nunca en el repositorio.               |
| RNF-03 | Cobertura de pruebas ≥ 80 % en las capas de dominio y aplicación.              |
| RNF-04 | Logs estructurados (JSON) con un identificador por petición.                   |
| RNF-05 | El proyecto completo se levanta con un solo comando (`docker compose up`).     |
| RNF-06 | Lint, tipado estricto y pruebas se ejecutan en CI en cada push y pull request. |

## 6. Criterio de "terminado" del proyecto

- Todos los RF tienen pruebas automatizadas que los cubren.
- El README tiene capturas o un GIF, instrucciones de uso y el último reporte de evaluación.
- Hay una demo accesible (despliegue público o video corto).
