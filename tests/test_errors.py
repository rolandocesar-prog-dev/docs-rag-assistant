from pathlib import Path

import pytest

from rag_assistant.domain.errors import UnsupportedFormatError


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path("data/informe.docx"), "informe.docx (.docx)"),
        (Path("data/LEEME"), "LEEME (sin extensión)"),
    ],
)
def test_unsupported_format_error_names_the_file_and_its_format(path: Path, expected: str) -> None:
    error = UnsupportedFormatError(path)

    assert error.path == path
    assert expected in str(error)
