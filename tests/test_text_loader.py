from pathlib import Path

from rag_assistant.adapters.loaders.text import TextLoader


def test_loads_plain_text_with_file_name_as_title(tmp_path: Path) -> None:
    path = tmp_path / "notas-de-version.txt"
    path.write_bytes("Versión 1.0  \r\n\r\n\r\nCambios menores.\r\n".encode())

    document = TextLoader().load(path, license="MIT")

    assert document.title == "notas-de-version"
    assert document.parts[0].text == "Versión 1.0\n\nCambios menores."
    assert document.parts[0].page is None
