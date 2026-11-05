from itertools import pairwise

import pytest

from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.chunking import split_document
from rag_assistant.domain.document import Document, DocumentPart


def document(*parts: DocumentPart) -> Document:
    return Document(source_path="doc.md", title="Doc", license="MIT", parts=parts)


def split(text: str, **sizes: int) -> list[Chunk]:
    return split_document(document(DocumentPart(text)), **sizes)


def words(count: int, prefix: str = "palabra") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_text_without_headings_is_a_single_chunk_without_section() -> None:
    doc = document(DocumentPart("Un Pod agrupa contenedores."))

    [chunk] = split_document(doc)

    assert (chunk.content, chunk.section, chunk.page, chunk.position) == (
        "Un Pod agrupa contenedores.",
        None,
        None,
        0,
    )
    assert chunk.document_id == doc.id


def test_sections_follow_the_heading_path() -> None:
    text = (
        "Intro.\n\n# Service\n\nDefinición.\n\n## Tipos\n\n### ClusterIP\n\nInterno.\n\n"
        "## Selectores\n\nEtiquetas."
    )

    chunks = split(text)

    assert [(c.section, c.content) for c in chunks] == [
        (None, "Intro."),
        ("Service", "# Service\n\nDefinición."),
        ("Service > Tipos > ClusterIP", "### ClusterIP\n\nInterno."),
        ("Service > Selectores", "## Selectores\n\nEtiquetas."),
    ]
    assert [c.position for c in chunks] == [0, 1, 2, 3]


def test_hashes_inside_code_blocks_are_not_headings() -> None:
    text = "## Ejemplo\n\n```yaml\n# comentario de YAML\nkind: Pod\n```\n\n~~~\n# otro\n~~~"

    [chunk] = split(text)

    assert chunk.section == "Ejemplo"
    assert "# comentario de YAML" in chunk.content
    assert chunk.content.endswith("~~~\n# otro\n~~~")


def test_long_section_is_split_within_limit_with_overlap_and_no_lost_words() -> None:
    paragraphs = [words(30, prefix=f"p{index}_") for index in range(12)]
    text = "## Larga\n\n" + "\n\n".join(paragraphs)

    chunks = split(text, max_chars=600, overlap=100)

    assert len(chunks) > 1
    assert all(len(c.content) <= 600 for c in chunks)
    assert all(c.section == "Larga" for c in chunks)
    joined = " ".join(c.content for c in chunks)
    assert all(word in joined.split() for word in text.split())
    for previous, following in pairwise(chunks):
        first_word = following.content.split()[0]
        assert first_word in previous.content.split()[-20:]


def test_paragraph_larger_than_limit_is_cut_by_lines_then_words() -> None:
    text = "\n".join(words(15, prefix=f"l{index}_") for index in range(10))

    chunks = split(text, max_chars=200, overlap=0)

    assert all(len(c.content) <= 200 for c in chunks)
    assert " ".join(" ".join(c.content.split()) for c in chunks) == " ".join(text.split())


def test_giant_word_is_cut_by_characters() -> None:
    giant = "x" * 1000

    chunks = split(giant, max_chars=300, overlap=0)

    assert all(len(c.content) <= 300 for c in chunks)
    assert "".join(c.content for c in chunks) == giant


def test_code_block_with_blank_lines_stays_in_one_chunk_when_it_fits() -> None:
    code = "```\nlinea 1\n\nlinea 2\n```"
    text = f"{words(40)}\n\n{code}\n\n{words(40, prefix='fin')}"

    chunks = split(text, max_chars=400, overlap=0)

    assert any(code in c.content for c in chunks)


def test_chunks_never_mix_pages_and_keep_page_numbers() -> None:
    doc = document(DocumentPart("Página uno.", page=1), DocumentPart("Página dos.", page=2))

    chunks = split_document(doc)

    assert [(c.page, c.content, c.position) for c in chunks] == [
        (1, "Página uno.", 0),
        (2, "Página dos.", 1),
    ]


def test_headings_without_body_produce_no_chunks() -> None:
    chunks = split("# Título\n\n## Vacía\n\n## Con texto\n\nContenido.")

    assert [(c.section, c.content) for c in chunks] == [
        ("Título > Con texto", "## Con texto\n\nContenido.")
    ]


def test_is_deterministic_except_for_ids() -> None:
    text = "## A\n\n" + words(300)

    first = split(text, max_chars=500, overlap=50)
    second = split(text, max_chars=500, overlap=50)

    assert [c.content for c in first] == [c.content for c in second]


@pytest.mark.parametrize(
    ("max_chars", "overlap"),
    [(100, -1), (100, 50), (100, 60), (2, 0)],
    ids=["negativo", "mitad", "mas-de-la-mitad", "maximo-diminuto"],
)
def test_rejects_invalid_sizes(max_chars: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        split("Texto.", max_chars=max_chars, overlap=overlap)
