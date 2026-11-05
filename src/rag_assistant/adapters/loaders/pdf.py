"""Cargador de PDF con texto (sin OCR): una parte por página con texto, conservando su número."""

from pathlib import Path

from pypdf import PdfReader

from rag_assistant.adapters.loaders._text import normalize
from rag_assistant.domain.document import Document, DocumentPart
from rag_assistant.domain.errors import UnsupportedFormatError


class PdfLoader:
    extensions = frozenset({".pdf"})

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def load(self, path: Path, *, license: str) -> Document:
        if not self.supports(path):
            raise UnsupportedFormatError(path)
        reader = PdfReader(path)
        parts = tuple(
            DocumentPart(text, page=number)
            for number, page in enumerate(reader.pages, start=1)
            if (text := normalize(page.extract_text()))
        )
        metadata_title = reader.metadata.title if reader.metadata else None
        return Document(
            source_path=path.as_posix(),
            title=(metadata_title or "").strip() or path.stem,
            license=license,
            parts=parts,
        )
