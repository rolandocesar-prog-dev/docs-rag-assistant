"""Documento fuente ya cargado y normalizado, listo para dividirse en fragmentos (RF-01)."""

import hashlib
import json
from dataclasses import dataclass, field
from uuid import UUID, uuid4

from rag_assistant.domain.errors import EmptyDocumentError


@dataclass(frozen=True, slots=True)
class DocumentPart:
    """Texto contiguo de un documento. `page` solo aplica a formatos paginados (PDF)."""

    text: str
    page: int | None = None

    def __post_init__(self) -> None:
        if self.page is not None and self.page < 1:
            raise ValueError("La página debe ser mayor o igual a 1.")


@dataclass(frozen=True, slots=True)
class Document:
    source_path: str
    title: str
    license: str
    parts: tuple[DocumentPart, ...]
    id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        for name in ("source_path", "title", "license"):
            if not getattr(self, name).strip():
                raise ValueError(f"El campo {name} no puede estar vacío.")
        if not any(part.text.strip() for part in self.parts):
            raise EmptyDocumentError(f"El documento {self.source_path} no tiene texto.")

    @property
    def content_hash(self) -> str:
        """SHA-256 del contenido (texto y páginas); no depende del título ni de la licencia."""
        payload = json.dumps([[part.page, part.text] for part in self.parts], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
