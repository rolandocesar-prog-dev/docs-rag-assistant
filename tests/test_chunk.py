from typing import Any
from uuid import uuid4

import pytest

from rag_assistant.domain.chunk import Chunk

DOCUMENT_ID = uuid4()


def test_keeps_its_metadata_and_gets_a_unique_id() -> None:
    first = Chunk(DOCUMENT_ID, position=0, content="Un Pod…", section="¿Qué es un Pod?", page=None)
    second = Chunk(DOCUMENT_ID, position=1, content="Los contenedores…")

    assert first.document_id == DOCUMENT_ID
    assert (first.position, first.section, first.page) == (0, "¿Qué es un Pod?", None)
    assert first.id != second.id


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"position": -1}, "posición"),
        ({"content": " \n"}, "no tiene texto"),
        ({"section": "  "}, "sección"),
        ({"page": 0}, "página"),
    ],
    ids=["posicion-negativa", "sin-texto", "seccion-vacia", "pagina-cero"],
)
def test_rejects_invalid_values(overrides: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {"position": 0, "content": "texto", "section": None, "page": None}
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        Chunk(DOCUMENT_ID, **values)
