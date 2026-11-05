"""Cargador de texto plano: el título es el nombre del archivo."""

from pathlib import Path

from rag_assistant.adapters.loaders._text import normalize, read_text
from rag_assistant.domain.document import Document, DocumentPart
from rag_assistant.domain.errors import UnsupportedFormatError


class TextLoader:
    extensions = frozenset({".txt"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def load(self, path: Path, *, license: str) -> Document:
        if not self.supports(path):
            raise UnsupportedFormatError(path)
        return Document(
            source_path=path.as_posix(),
            title=path.stem,
            license=license,
            parts=(DocumentPart(normalize(read_text(path))),),
        )
