"""División de documentos en fragmentos: por encabezados y luego por tamaño (RF-01, design.md)."""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.document import Document

# ≈ 800 y 100 tokens, con unos 4 caracteres por token.
DEFAULT_MAX_CHARS = 3200
DEFAULT_OVERLAP = 400

_PARAGRAPH = "\n\n"
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
# Para cortar un párrafo demasiado grande: primero por líneas, luego por palabras.
_FALLBACK_SEPARATORS = ("\n", " ")


@dataclass(frozen=True, slots=True)
class _Section:
    path: str | None
    text: str


def split_document(
    document: Document,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    if overlap < 0 or 2 * overlap >= max_chars:
        raise ValueError("El solapamiento debe ser >= 0 y menor que la mitad del máximo.")
    if max_chars - overlap - len(_PARAGRAPH) < 1:
        raise ValueError("El tamaño máximo es demasiado pequeño para ese solapamiento.")

    chunks: list[Chunk] = []
    for part in document.parts:
        for section in _sections(part.text):
            for content in _split_by_size(section.text, max_chars, overlap):
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        position=len(chunks),
                        content=content,
                        section=section.path,
                        page=part.page,
                    )
                )
    return chunks


def _sections(text: str) -> Iterator[_Section]:
    """Agrupa las líneas bajo su encabezado, ignorando los `#` de los bloques de código."""
    headings: list[tuple[int, str]] = []
    path: str | None = None
    lines: list[str] = []
    has_body = False
    fence: str | None = None

    for line in text.split("\n"):
        fence, is_fence = _track_fence(line, fence)
        heading = _HEADING.match(line) if fence is None and not is_fence else None
        if heading is None:
            lines.append(line)
            has_body = has_body or bool(line.strip())
            continue
        if has_body:
            yield _Section(path, "\n".join(lines).strip())
        level = len(heading.group(1))
        headings = [(lvl, title) for lvl, title in headings if lvl < level]
        headings.append((level, heading.group(2)))
        path = " > ".join(title for _, title in headings)
        lines, has_body = [line], False

    if has_body:
        yield _Section(path, "\n".join(lines).strip())


def _track_fence(line: str, fence: str | None) -> tuple[str | None, bool]:
    """Estado del bloque de código (```/~~~) después de `line`, y si `line` es una valla."""
    match = _FENCE.match(line)
    if match is None:
        return fence, False
    marker = match.group(1)
    if fence is None:
        return marker[0] * 3, True
    return (None if marker.startswith(fence) else fence), True


def _split_by_size(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    capacity = max_chars - overlap - len(_PARAGRAPH)
    pieces: list[str] = []
    current = ""
    for unit in (u for block in _blocks(text) for u in _fit(block, capacity)):
        candidate = f"{current}{_PARAGRAPH}{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        pieces.append(current)
        tail = _tail(current, overlap)
        current = f"{tail}{_PARAGRAPH}{unit}" if tail else unit
    pieces.append(current)
    return [piece for piece in pieces if piece.strip()]


def _blocks(text: str) -> list[str]:
    """Párrafos separados por líneas en blanco; un bloque de código cuenta como uno solo."""
    blocks: list[str] = []
    current: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        fence, _ = _track_fence(line, fence)
        if not line.strip() and fence is None:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _fit(text: str, capacity: int, level: int = 0) -> list[str]:
    """Corta `text` en trozos de hasta `capacity` caracteres, por líneas, palabras o caracteres."""
    if len(text) <= capacity:
        return [text]
    if level == len(_FALLBACK_SEPARATORS):
        return [text[start : start + capacity] for start in range(0, len(text), capacity)]
    separator = _FALLBACK_SEPARATORS[level]
    pieces: list[str] = []
    current = ""
    for piece in text.split(separator):
        for sub in _fit(piece, capacity, level + 1):
            candidate = f"{current}{separator}{sub}" if current else sub
            if len(candidate) <= capacity:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = sub
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap: int) -> str:
    """Final de `text` de hasta `overlap` caracteres, empezando en un límite de palabra."""
    if overlap == 0:
        return ""
    tail = text[-overlap:]
    if len(text) > overlap and not text[-overlap - 1].isspace():
        boundary = re.search(r"\s", tail)
        if boundary is not None:
            tail = tail[boundary.end() :]
    return tail.strip()
