from uuid import uuid4

import pytest

from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.prompting import (
    NOT_FOUND,
    SYSTEM_PROMPT,
    build_prompt,
    excerpt,
    parse_answer,
)
from rag_assistant.domain.retrieval import RetrievedChunk


def retrieved(
    content: str, section: str | None, *, title: str = "Pods", page: int | None = None
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(uuid4(), 0, content, section=section, page=page),
        document_title=title,
        source_path=f"concepts/{title.lower()}.md",
        score=0.7,
    )


CONTEXT = [
    retrieved("## ¿Qué es un Pod?\n\nUn Pod agrupa contenedores.", "Pods > ¿Qué es un Pod?"),
    retrieved("Los Pods comparten red y almacenamiento.", None),
    retrieved("Un Deployment gestiona réplicas.", "Deployment", title="Deployment", page=3),
]


def test_system_prompt_asks_for_citations_and_the_exact_not_found_phrase() -> None:
    assert "[1]" in SYSTEM_PROMPT
    assert NOT_FOUND in SYSTEM_PROMPT


def test_prompt_numbers_fragments_with_title_and_section() -> None:
    prompt = build_prompt("¿Qué es un Pod?", CONTEXT)

    assert prompt.startswith(
        "Fragmentos:\n\n[1] Documento: Pods\nSección: Pods > ¿Qué es un Pod?\n## ¿Qué es"
    )
    assert "\n\n[2] Documento: Pods\nLos Pods comparten" in prompt
    assert "\n\n[3] Documento: Deployment\nSección: Deployment\nUn Deployment" in prompt
    assert prompt.endswith("\n\nPregunta: ¿Qué es un Pod?")


def test_valid_answer_keeps_cited_sources_in_order() -> None:
    answer = parse_answer("Un Pod agrupa contenedores [1] que comparten red [2][1].", CONTEXT)

    assert answer.found
    assert answer.text == "Un Pod agrupa contenedores [1] que comparten red [2][1]."
    assert [c.id for c in answer.citations] == [1, 2]
    first = answer.citations[0]
    assert (first.document, first.section, first.source_path) == (
        "Pods",
        "Pods > ¿Qué es un Pod?",
        "concepts/pods.md",
    )
    assert first.excerpt == "Un Pod agrupa contenedores."


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_ids"),
    [
        ("Gestiona réplicas [3, 1].", "Gestiona réplicas [3][1].", [1, 3]),
        ("Gestiona réplicas [3][7].", "Gestiona réplicas [3].", [3]),
        ("Agrupa contenedores [0] y réplicas [3].", "Agrupa contenedores y réplicas [3].", [3]),
        ("Agrupa contenedores [ 2 , 9 ].", "Agrupa contenedores [2].", [2]),
    ],
    ids=["coma", "inexistente", "cero", "espacios"],
)
def test_invalid_citation_numbers_are_removed(
    text: str, expected_text: str, expected_ids: list[int]
) -> None:
    answer = parse_answer(text, CONTEXT)

    assert answer.text == expected_text
    assert [c.id for c in answer.citations] == expected_ids


@pytest.mark.parametrize(
    "text",
    [
        NOT_FOUND,
        "Un Pod agrupa contenedores, sin citar nada.",
        "Según el fragmento [8], un Pod agrupa contenedores.",
        "",
    ],
    ids=["rechazo", "sin-citas", "solo-citas-invalidas", "vacia"],
)
def test_answers_without_valid_citations_become_not_found(text: str) -> None:
    answer = parse_answer(text, CONTEXT)

    assert (answer.found, answer.text, answer.citations) == (False, NOT_FOUND, ())


def test_not_found_phrase_is_dropped_when_there_are_valid_citations() -> None:
    answer = parse_answer(f"Un Pod agrupa contenedores [1].\n\n{NOT_FOUND}", CONTEXT)

    assert answer.found
    assert answer.text == "Un Pod agrupa contenedores [1]."


def test_page_is_kept_for_paginated_sources() -> None:
    [citation] = parse_answer("Gestiona réplicas [3].", CONTEXT).citations

    assert citation.page == 3


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("# Título\n\nTexto   con\nsaltos.", "Texto con saltos."),
        ("Un [objeto](/docs/concepts/objeto/) de la API.", "Un objeto de la API."),
        ("palabra " * 40, "palabra " * 24 + "palabra…"),
        ("x" * 250, "x" * 200 + "…"),
    ],
    ids=["sin-encabezados", "enlace", "corte-en-palabra", "sin-espacios"],
)
def test_excerpt(content: str, expected: str) -> None:
    assert excerpt(content) == expected


@pytest.mark.parametrize(
    "text",
    [
        "La documentación proporcionada no contiene instrucciones sobre cómo instalar Python [4].",
        "Los fragmentos no mencionan cómo guardar contraseñas; solo hablan de tokens [1].",
        "El contexto no explica eso [2]. Sí describe los Pods [1].",
        "No encontré información sobre eso en la documentación [1].",
    ],
    ids=["no-contiene", "no-mencionan", "no-explica", "no-encontre"],
)
def test_paraphrased_refusals_with_citations_become_not_found(text: str) -> None:
    answer = parse_answer(text, CONTEXT)

    assert (answer.found, answer.text, answer.citations) == (False, NOT_FOUND, ())


@pytest.mark.parametrize(
    "text",
    [
        "Un ConfigMap no proporciona encriptación [1].",
        "Un Pod agrupa contenedores [1]. La documentación no menciona su límite de tamaño.",
        "Según la documentación, un Deployment gestiona réplicas [3].",
    ],
    ids=["negacion-sin-fuentes", "limite-en-segunda-frase", "afirmacion-sobre-fuentes"],
)
def test_answers_that_only_mention_limits_are_kept(text: str) -> None:
    assert parse_answer(text, CONTEXT).found
