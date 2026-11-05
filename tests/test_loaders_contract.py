"""Comportamiento común a todos los cargadores; el tipo de LOADERS verifica el puerto con mypy."""

from pathlib import Path

import pytest

from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.adapters.loaders.pdf import PdfLoader
from rag_assistant.adapters.loaders.text import TextLoader
from rag_assistant.domain.errors import UnsupportedFormatError
from rag_assistant.ports.document_loader import DocumentLoader

LOADERS: list[DocumentLoader] = [MarkdownLoader(), TextLoader(), PdfLoader()]


@pytest.mark.parametrize("loader", LOADERS, ids=lambda loader: type(loader).__name__)
def test_rejects_unsupported_formats(loader: DocumentLoader, tmp_path: Path) -> None:
    path = tmp_path / "informe.docx"
    path.write_bytes(b"no soportado")

    assert not loader.supports(path)
    with pytest.raises(UnsupportedFormatError):
        loader.load(path, license="MIT")


def test_each_extension_has_exactly_one_loader() -> None:
    for name in ["a.md", "a.markdown", "a.txt", "a.pdf"]:
        assert sum(loader.supports(Path(name)) for loader in LOADERS) == 1
