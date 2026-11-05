"""Puerto de carga de documentos: un adaptador por formato (Markdown, TXT, PDF)."""

from pathlib import Path
from typing import Protocol

from rag_assistant.domain.document import Document


class DocumentLoader(Protocol):
    def supports(self, path: Path) -> bool:
        """Indica si este cargador sabe leer el archivo, según su formato."""
        ...

    def load(self, path: Path, *, license: str) -> Document:
        """Lee y normaliza el archivo.

        Lanza `UnsupportedFormatError` si el formato no corresponde a este cargador.
        """
        ...
