"""Métricas, lectura del conjunto y veredictos del juez (lógica pura de RF-06)."""

import json
from uuid import uuid4

import pytest

from rag_assistant.domain.answer import Answer, Citation
from rag_assistant.domain.chunk import Chunk
from rag_assistant.domain.evaluation import (
    EvalItem,
    ExpectedSource,
    Verdict,
    build_judge_prompt,
    cites_expected_source,
    first_relevant_rank,
    hit_rate,
    mean_reciprocal_rank,
    parse_dataset,
    parse_verdict,
)
from rag_assistant.domain.retrieval import RetrievedChunk

ITEM = EvalItem(
    id="q01",
    category="definicion",
    question="¿Qué es un Pod?",
    expected=(ExpectedSource("concepts/pod.md", "¿Qué es?"),),
    reference="Un Pod agrupa contenedores.",
)


def result(path: str, section: str | None) -> RetrievedChunk:
    return RetrievedChunk(Chunk(uuid4(), 0, "texto", section=section), "Doc", path, 0.6)


def line(**overrides: object) -> str:
    raw: dict[str, object] = {
        "id": "q01",
        "category": "definicion",
        "question": "¿Qué es un Pod?",
        "expected": [{"source_path": "concepts/pod.md", "section": None}],
        "reference": "Un Pod agrupa contenedores.",
    }
    return json.dumps(raw | overrides, ensure_ascii=False)


def test_parses_items_and_skips_blank_lines() -> None:
    off_topic = line(id="q02", category="fuera_de_tema", expected=[], reference=None)

    first, second = parse_dataset([line(), "", off_topic])

    assert first.expected == (ExpectedSource("concepts/pod.md", None),)
    assert first.on_topic and not second.on_topic


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ("{no es json", "Línea 2"),
        (line(category="otra"), "categoría desconocida"),
        (line(expected=[]), "fuentes esperadas"),
        (line(category="fuera_de_tema"), "fuentes esperadas"),
        (json.dumps({"id": "q09"}), "Línea 2"),
    ],
    ids=["json", "categoria", "sin-fuentes", "fuera-de-tema-con-fuentes", "faltan-campos"],
)
def test_invalid_lines_report_their_number(bad: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_dataset([line(), bad])


def test_first_relevant_rank_by_document_and_by_section() -> None:
    results = [
        result("concepts/otro.md", "¿Qué es?"),
        result("concepts/pod.md", "Otra sección"),
        result("concepts/pod.md", "¿Qué es?"),
    ]

    assert first_relevant_rank(results, ITEM, by_section=False) == 2
    assert first_relevant_rank(results, ITEM, by_section=True) == 3
    assert first_relevant_rank(results[:1], ITEM, by_section=False) is None


def test_hit_rate_and_mrr() -> None:
    ranks = [1, 3, None]

    assert hit_rate(ranks, 1) == pytest.approx(1 / 3)
    assert hit_rate(ranks, 4) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank(ranks) == pytest.approx((1 + 1 / 3) / 3)
    assert hit_rate([], 4) == 0.0 and mean_reciprocal_rank([]) == 0.0


def test_cites_expected_source() -> None:
    def answer(path: str) -> Answer:
        return Answer("Texto [1].", (Citation(1, "Doc", None, None, path, "..."),), found=True)

    assert cites_expected_source(answer("concepts/pod.md"), ITEM)
    assert not cites_expected_source(answer("concepts/otro.md"), ITEM)


def test_judge_prompt_includes_question_fragments_answer_and_reference() -> None:
    answer = Answer("Un Pod agrupa contenedores [2] en un nodo [4].", (), found=True)

    prompt = build_judge_prompt(
        ITEM, answer, [(2, "Un Pod es un grupo de contenedores."), (4, "Un nodo es una máquina.")]
    )

    for part in (
        "¿Qué es un Pod?",
        "[2] Un Pod es un grupo",
        "[4] Un nodo es una máquina.",
        "agrupa contenedores [2]",
        "Un Pod agrupa contenedores.",
    ):
        assert part in prompt
    assert "[1]" not in prompt
    assert "(ninguno)" in build_judge_prompt(ITEM, answer, [])


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"fiel": true, "correcta": false}', Verdict(faithful=True, correct=False)),
        ('Veredicto:\n```json\n{"fiel": false, "correcta": true}\n```', Verdict(False, True)),
        ('{"fiel": "sí", "correcta": true}', None),
        ('{"fiel": true}', None),
        ("{no es json}", None),
        ("[true, false]", None),
        ("sin json", None),
    ],
    ids=["valido", "con-texto", "no-booleano", "falta-campo", "json-roto", "lista", "sin-json"],
)
def test_parse_verdict(text: str, expected: Verdict | None) -> None:
    assert parse_verdict(text) == expected
