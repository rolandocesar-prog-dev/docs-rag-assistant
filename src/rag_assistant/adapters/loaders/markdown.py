"""Cargador de Markdown: quita el bloque YAML, comentarios HTML, anclas y etiquetas de Hugo."""

import re
from pathlib import Path

from rag_assistant.adapters.loaders._text import normalize, read_text
from rag_assistant.adapters.loaders.hugo import strip_hugo
from rag_assistant.domain.document import Document, DocumentPart
from rag_assistant.domain.errors import UnsupportedFormatError
from rag_assistant.domain.markdown import track_fence

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---[ \t]*(?:\n|\Z)", re.DOTALL)
_TITLE = re.compile(r"^title:[ \t]*(.+?)[ \t]*$", re.MULTILINE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
# Ancla de un encabezado: `## Título {#titulo}` → `## Título`.
_HEADING_ANCHOR = re.compile(r"^(#{1,6}[ \t]+.*?)[ \t]*\{#[^}]*\}[ \t]*$")


class MarkdownLoader:
    extensions = frozenset({".md", ".markdown"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def load(self, path: Path, *, license: str) -> Document:
        if not self.supports(path):
            raise UnsupportedFormatError(path)
        front_matter_title, body = _split_front_matter(read_text(path))
        text = normalize(_strip_heading_anchors(strip_hugo(_HTML_COMMENT.sub("", body))))
        heading = _HEADING.search(text)
        title = front_matter_title or (heading.group(1) if heading else None) or path.stem
        return Document(
            source_path=path.as_posix(),
            title=title,
            license=license,
            parts=(DocumentPart(text),),
        )


def _split_front_matter(text: str) -> tuple[str | None, str]:
    """Separa el bloque YAML inicial y devuelve su `title` (si lo hay) y el resto del texto."""
    match = _FRONT_MATTER.match(text)
    if match is None:
        return None, text
    title = _TITLE.search(match.group(1))
    return (_unquote(title.group(1)) if title else None), text[match.end() :]


def _strip_heading_anchors(text: str) -> str:
    """Quita `{#id}` de los encabezados, sin tocar los `#` de los bloques de código."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        fence, is_fence = track_fence(line, fence)
        if fence is None and not is_fence:
            line = _HEADING_ANCHOR.sub(r"\1", line)
        lines.append(line)
    return "\n".join(lines)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].strip()
    return value
