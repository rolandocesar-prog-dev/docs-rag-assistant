"""El corpus versionado se carga completo y sin restos de Hugo ni HTML (data/SOURCES.md)."""

from functools import cache
from pathlib import Path

import pytest

from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.domain.document import Document
from rag_assistant.domain.errors import EmptyDocumentError

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus" / "kubernetes" / "concepts"
FILES = sorted(CORPUS.rglob("*.md"))


@cache
def load_corpus() -> tuple[list[Document], list[Path]]:
    loader = MarkdownLoader()
    loaded: list[Document] = []
    empty: list[Path] = []
    for path in FILES:
        try:
            loaded.append(loader.load(path, license="CC-BY-4.0"))
        except EmptyDocumentError:
            empty.append(path)
    return loaded, empty


def test_corpus_is_present() -> None:
    assert len(FILES) == 87


def test_only_section_indexes_can_be_empty() -> None:
    loaded, empty = load_corpus()

    assert {path.name for path in empty} <= {"_index.md"}
    assert len(loaded) >= 70


@pytest.mark.parametrize("leftover", ["{{<", "{{%", ">}}", "%}}", "<!--"])
def test_no_markup_leftovers(leftover: str) -> None:
    loaded, _ = load_corpus()

    assert [doc.source_path for doc in loaded if leftover in doc.parts[0].text] == []


def test_every_document_has_a_title() -> None:
    loaded, _ = load_corpus()

    assert all(doc.title and doc.title != Path(doc.source_path).stem for doc in loaded)
