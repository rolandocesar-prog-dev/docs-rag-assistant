from pathlib import Path

import pytest

from rag_assistant.adapters.loaders.pdf import PdfLoader
from rag_assistant.domain.errors import EmptyDocumentError

LOADER = PdfLoader()


def write_pdf(path: Path, pages: list[str | None], title: str | None = None) -> Path:
    """Escribe un PDF mínimo con una línea de texto ASCII por página (`None` = página vacía)."""
    first_page_id = 5
    page_ids = [first_page_id + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Title ({title or ''}) >>".encode(),
    ]
    for page_id, text in zip(page_ids, pages, strict=True):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode() if text else b""
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content), content))

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_offset = len(pdf)
    pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    pdf += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    info = " /Info 4 0 R" if title else ""
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R{info} >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode()
    path.write_bytes(bytes(pdf))
    return path


def test_one_part_per_page_skipping_pages_without_text(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "guia.pdf", ["Primera pagina", None, "Tercera pagina"])

    document = LOADER.load(path, license="MIT")

    assert [(part.page, part.text) for part in document.parts] == [
        (1, "Primera pagina"),
        (3, "Tercera pagina"),
    ]


@pytest.mark.parametrize(("title", "expected"), [("Manual", "Manual"), (None, "guia")])
def test_title_comes_from_metadata_or_file_name(
    tmp_path: Path, title: str | None, expected: str
) -> None:
    path = write_pdf(tmp_path / "guia.pdf", ["Texto"], title=title)

    assert LOADER.load(path, license="MIT").title == expected


def test_pdf_without_text_is_empty(tmp_path: Path) -> None:
    path = write_pdf(tmp_path / "escaneado.pdf", [None, None])

    with pytest.raises(EmptyDocumentError):
        LOADER.load(path, license="MIT")
