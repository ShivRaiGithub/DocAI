import pytest

from app.extraction import ExtractionError, extract_pages


def make_pdf(*page_texts: str) -> bytes:
    """Build a tiny valid PDF by hand, one text line per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(page_texts)))
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_texts)} >>")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(page_texts):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for n, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_txt_and_md_are_single_page():
    pages = extract_pages("héllo\nworld".encode(), ".txt")
    assert [(p.number, p.text) for p in pages] == [(1, "héllo\nworld")]
    assert extract_pages(b"# Title", ".md")[0].text == "# Title"


def test_bom_stripped_and_latin1_fallback():
    assert extract_pages(b"\xef\xbb\xbfhi", ".txt")[0].text == "hi"
    assert extract_pages(b"caf\xe9", ".txt")[0].text == "café"


def test_pdf_pages_are_numbered():
    pages = extract_pages(make_pdf("First page", "Second page"), ".pdf")
    assert [(p.number, p.text) for p in pages] == [(1, "First page"), (2, "Second page")]


def test_empty_text_corrupt_pdf_and_unknown_type_raise():
    with pytest.raises(ExtractionError):
        extract_pages(b"   \n ", ".txt")
    with pytest.raises(ExtractionError):
        extract_pages(b"not a pdf", ".pdf")
    with pytest.raises(ExtractionError):
        extract_pages(b"x", ".exe")
