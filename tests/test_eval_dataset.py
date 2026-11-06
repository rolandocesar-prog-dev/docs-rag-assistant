"""El conjunto de evaluación (RF-06) es válido y apunta a documentos y secciones que existen."""

import json
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from rag_assistant.adapters.loaders.markdown import MarkdownLoader
from rag_assistant.domain.chunking import split_document

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval" / "dataset.jsonl"
# `source_path` es relativo a la carpeta padre de la ingestada (data/corpus/kubernetes/concepts).
CORPUS_PARENT = ROOT / "data" / "corpus" / "kubernetes"
CATEGORIES = {"definicion", "procedimiento", "comparacion", "parafrasis", "fuera_de_tema"}


def load() -> list[dict[str, object]]:
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@cache
def sections(source_path: str) -> frozenset[str | None]:
    document = MarkdownLoader().load(CORPUS_PARENT / source_path, license="CC-BY-4.0")
    return frozenset(chunk.section for chunk in split_document(document))


def test_has_at_least_20_on_topic_questions_and_some_off_topic() -> None:
    categories = Counter(str(item["category"]) for item in load())

    assert set(categories) <= CATEGORIES
    assert sum(n for c, n in categories.items() if c != "fuera_de_tema") >= 20
    assert categories["fuera_de_tema"] >= 3


def test_ids_and_questions_are_unique() -> None:
    items = load()

    assert len({item["id"] for item in items}) == len(items)
    assert len({item["question"] for item in items}) == len(items)


@pytest.mark.parametrize("item", load(), ids=lambda item: str(item["id"]))
def test_item_is_consistent(item: dict[str, object]) -> None:
    assert set(item) == {"id", "category", "question", "expected", "reference"}
    question, expected, reference = item["question"], item["expected"], item["reference"]
    assert isinstance(question, str) and 0 < len(question) <= 1000
    assert isinstance(expected, list)
    if item["category"] == "fuera_de_tema":
        assert expected == [] and reference is None
        return
    assert expected and isinstance(reference, str) and reference.strip()
    for target in expected:
        source_path, section = target["source_path"], target["section"]
        assert (CORPUS_PARENT / source_path).is_file(), source_path
        assert section in sections(source_path), f"{source_path} no tiene la sección {section!r}"
