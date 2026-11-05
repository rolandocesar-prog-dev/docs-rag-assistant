from dataclasses import FrozenInstanceError

import pytest

from rag_assistant.domain.document import Document, DocumentPart


def make_document(
    *parts: DocumentPart, title: str = "Pods", license: str = "CC-BY-4.0"
) -> Document:
    return Document(
        source_path="kubernetes/concepts/workloads/pods/pod.md",
        title=title,
        license=license,
        parts=parts or (DocumentPart("Un Pod es la unidad mínima de despliegue."),),
    )


def test_content_hash_is_sha256_hex() -> None:
    document = make_document()

    assert len(document.content_hash) == 64
    assert all(char in "0123456789abcdef" for char in document.content_hash)


def test_content_hash_ignores_id_title_and_license() -> None:
    first = make_document(title="Pods", license="CC-BY-4.0")
    second = make_document(title="Otro título", license="MIT")

    assert first.id != second.id
    assert first.content_hash == second.content_hash


@pytest.mark.parametrize(
    "changed",
    [
        DocumentPart("Un Pod es la unidad mínima de despliegue!"),
        DocumentPart("Un Pod es la unidad mínima de despliegue.", page=2),
    ],
    ids=["texto", "pagina"],
)
def test_content_hash_changes_with_text_or_page(changed: DocumentPart) -> None:
    original = make_document(DocumentPart("Un Pod es la unidad mínima de despliegue.", page=1))

    assert make_document(changed).content_hash != original.content_hash


def test_content_hash_distinguishes_how_text_is_split_into_parts() -> None:
    joined = make_document(DocumentPart("ab"))
    split = make_document(DocumentPart("a"), DocumentPart("b"))

    assert joined.content_hash != split.content_hash


@pytest.mark.parametrize("field_name", ["source_path", "title", "license"])
def test_rejects_blank_metadata(field_name: str) -> None:
    values = {"source_path": "doc.md", "title": "Título", "license": "MIT"}
    values[field_name] = "   "

    with pytest.raises(ValueError, match=field_name):
        Document(
            source_path=values["source_path"],
            title=values["title"],
            license=values["license"],
            parts=(DocumentPart("texto"),),
        )


@pytest.mark.parametrize(
    "parts",
    [(), (DocumentPart(""), DocumentPart(" \n\t"))],
    ids=["sin-partes", "solo-espacios"],
)
def test_rejects_document_without_text(parts: tuple[DocumentPart, ...]) -> None:
    with pytest.raises(ValueError, match="no tiene texto"):
        Document(source_path="doc.md", title="Título", license="MIT", parts=parts)


def test_part_rejects_page_lower_than_one() -> None:
    with pytest.raises(ValueError, match="página"):
        DocumentPart("texto", page=0)


def test_document_is_immutable() -> None:
    document = make_document()

    with pytest.raises(FrozenInstanceError):
        document.title = "Otro"  # type: ignore[misc]
