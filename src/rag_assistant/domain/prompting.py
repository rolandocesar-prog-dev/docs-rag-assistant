"""Prompt con fragmentos numerados y validación de las citas de la respuesta (RF-02, RF-03)."""

import re
from collections.abc import Sequence

from rag_assistant.domain.answer import Answer, Citation
from rag_assistant.domain.retrieval import RetrievedChunk

NOT_FOUND = "No encontré información sobre eso en la documentación."
SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre la documentación de Kubernetes. "
    "Responde en español, en 2 a 5 frases, usando SOLO la información de los fragmentos "
    "numerados. Cita cada afirmación con el número del fragmento entre corchetes, por ejemplo "
    "[1] o [2][3]. Si los fragmentos no contienen la respuesta, responde exactamente: "
    f"«{NOT_FOUND}»"
)
EXCERPT_CHARS = 200

# `[2]`, `[1, 3]`: una marca con uno o más números separados por comas.
_CITATION = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")
_HEADING_LINE = re.compile(r"^#{1,6}[ \t].*$", re.MULTILINE)
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_NOT_FOUND_PHRASE = re.compile(re.escape(NOT_FOUND.rstrip(".")) + r"\.?", re.IGNORECASE)
# Rechazo con otras palabras: la primera frase habla de las fuentes y niega que cubran el tema.
_FIRST_SENTENCE = re.compile(r"^.*?(?:[.!?](?=\s|$)|\n|$)", re.DOTALL)
_SOURCES = re.compile(r"\b(?:documentaci[oó]n|fragmentos?|contexto)\b", re.IGNORECASE)
_NOT_COVERED = re.compile(
    r"\bno\s+(?:se\s+)?(?:contienen?|incluyen?|mencionan?|proporcionan?|ofrecen?|describen?|"
    r"explican?|tratan?|abordan?|encontr[eé])\b",
    re.IGNORECASE,
)


def not_found_answer() -> Answer:
    return Answer(text=NOT_FOUND, citations=(), found=False)


def build_prompt(question: str, retrieved: Sequence[RetrievedChunk]) -> str:
    """Mensaje de usuario: fragmentos numerados desde 1, con título y sección, y la pregunta."""
    blocks = [
        f"[{number}] Documento: {item.document_title}\n"
        + (f"Sección: {item.chunk.section}\n" if item.chunk.section else "")
        + item.chunk.content
        for number, item in enumerate(retrieved, start=1)
    ]
    return "Fragmentos:\n\n" + "\n\n".join(blocks) + f"\n\nPregunta: {question}"


def parse_answer(text: str, retrieved: Sequence[RetrievedChunk]) -> Answer:
    """Valida las citas del modelo. Sin ninguna cita válida, la respuesta no está fundamentada."""
    cited: set[int] = set()

    def keep_valid(match: re.Match[str]) -> str:
        numbers = [int(n) for n in match.group(1).split(",")]
        valid = [n for n in numbers if 1 <= n <= len(retrieved)]
        cited.update(valid)
        return "".join(f"[{n}]" for n in valid)

    cleaned = _CITATION.sub(keep_valid, text)
    if not cited or is_refusal(cleaned):
        return not_found_answer()
    cleaned = _NOT_FOUND_PHRASE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    citations = tuple(_citation(number, retrieved[number - 1]) for number in sorted(cited))
    return Answer(text=cleaned, citations=citations, found=True)


def is_refusal(text: str) -> bool:
    """La primera frase dice que las fuentes no cubren la pregunta (p. ej. «no contiene…»)."""
    first = _FIRST_SENTENCE.match(text.strip())
    sentence = first.group(0) if first else ""
    return bool(_SOURCES.search(sentence) and _NOT_COVERED.search(sentence))


def _citation(number: int, item: RetrievedChunk) -> Citation:
    return Citation(
        id=number,
        document=item.document_title,
        section=item.chunk.section,
        page=item.chunk.page,
        source_path=item.source_path,
        excerpt=excerpt(item.chunk.content),
    )


def excerpt(content: str, limit: int = EXCERPT_CHARS) -> str:
    """Comienzo del fragmento sin encabezados ni URL de enlaces, cortado en un límite de palabra."""
    flat = " ".join(_MARKDOWN_LINK.sub(r"\1", _HEADING_LINE.sub("", content)).split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.rstrip(" ,.;:") + "…"
