from pathlib import Path

import pytest

from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.domain.errors import EmptyDocumentError

LOADER = MarkdownLoader()


def write(tmp_path: Path, content: str, name: str = "pod.md") -> Path:
    path = tmp_path / name
    path.write_bytes(content.encode("utf-8"))
    return path


@pytest.mark.parametrize(
    ("name", "expected"), [("a.md", True), ("a.MARKDOWN", True), ("a.txt", False)]
)
def test_supports_markdown_extensions(name: str, expected: bool) -> None:
    assert LOADER.supports(Path(name)) is expected


def test_uses_front_matter_title_and_removes_the_block(tmp_path: Path) -> None:
    path = write(tmp_path, '---\ntitle: "Pods"\nweight: 10\n---\n\nUn Pod es la unidad mínima.\n')

    document = LOADER.load(path, license="CC-BY-4.0")

    assert document.title == "Pods"
    assert document.parts[0].text == "Un Pod es la unidad mínima."
    assert document.parts[0].page is None
    assert document.license == "CC-BY-4.0"
    assert document.source_path == path.as_posix()


def test_handles_bom_crlf_and_single_quoted_title(tmp_path: Path) -> None:
    path = write(tmp_path, "﻿---\r\ntitle: 'Nodos'\r\n---\r\nLínea 1\r\nLínea 2\r\n")

    document = LOADER.load(path, license="MIT")

    assert document.title == "Nodos"
    assert document.parts[0].text == "Línea 1\nLínea 2"


@pytest.mark.parametrize(
    ("content", "expected_title"),
    [
        ("Intro.\n\n## Instalación ##\n\nPasos.", "Instalación"),
        ("---\nweight: 1\n---\n# Guía\n\nTexto.", "Guía"),
        ("Solo texto, sin encabezados.", "pod"),
    ],
    ids=["encabezado", "yaml-sin-title", "nombre-de-archivo"],
)
def test_title_falls_back_to_first_heading_then_file_name(
    tmp_path: Path, content: str, expected_title: str
) -> None:
    assert LOADER.load(write(tmp_path, content), license="MIT").title == expected_title


def test_cleans_html_comments_hugo_tags_and_blank_lines(tmp_path: Path) -> None:
    content = (
        '---\ntitle: "Pods"\n---\n<!-- overview -->\n'
        'Un {{< glossary_tooltip text="Pod" term_id="pod" >}} agrupa contenedores.   \n\n\n\n'
        "{{< note >}}\nComparten red.\n{{< /note >}}\n\n"
        '## {{% heading "whatsnext" %}}\n'
    )

    text = LOADER.load(write(tmp_path, content), license="MIT").parts[0].text

    assert text == "Un Pod agrupa contenedores.\n\nComparten red.\n\n## ¿Qué sigue?"


def test_removes_heading_anchors_but_not_braces_elsewhere(tmp_path: Path) -> None:
    content = (
        "## Componentes del Clúster {#cluster-components}\n\n"
        "### Recuperación {##pod-fault-recovery}  \n\n"
        "#### Configuración {#configuración}\n\n"
        "El texto {#no-es-ancla} queda igual.\n\n"
        "```yaml\n# comentario {#tampoco}\n```"
    )

    text = LOADER.load(write(tmp_path, content), license="MIT").parts[0].text

    assert text == (
        "## Componentes del Clúster\n\n"
        "### Recuperación\n\n"
        "#### Configuración\n\n"
        "El texto {#no-es-ancla} queda igual.\n\n"
        "```yaml\n# comentario {#tampoco}\n```"
    )


def test_document_with_only_metadata_is_empty(tmp_path: Path) -> None:
    path = write(tmp_path, '---\ntitle: "Pods"\nweight: 10\n---\n<!-- body -->\n', "_index.md")

    with pytest.raises(EmptyDocumentError):
        LOADER.load(path, license="MIT")
