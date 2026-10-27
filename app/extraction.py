from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.schemas import ExtractedPage


class ExtractionError(Exception):
    """The file could not be turned into text (corrupt, encrypted, or no text layer)."""


def extract_pages(data: bytes, ext: str) -> list[ExtractedPage]:
    """Turn file bytes into a list of pages of text.

    Plain-text formats are a single page. PDFs give one entry per page, which
    lets later stages cite "page 3" in answers.
    """
    if ext == ".pdf":
        pages = _extract_pdf(data)
    elif ext in (".txt", ".md"):
        pages = [ExtractedPage(number=1, text=_decode(data))]
    else:
        raise ExtractionError(f"No extractor for '{ext}' files")

    if not any(p.text.strip() for p in pages):
        raise ExtractionError(
            "No text found. The file may be empty or a scanned image (OCR is not supported)."
        )
    return pages


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")  # also strips a Windows BOM if present
    except UnicodeDecodeError:
        return data.decode("latin-1")  # never fails; best effort for legacy files


def _extract_pdf(data: bytes) -> list[ExtractedPage]:
    try:
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ExtractionError("PDF is password-protected")
        return [
            ExtractedPage(number=i, text=(page.extract_text() or "").strip())
            for i, page in enumerate(reader.pages, start=1)
        ]
    except PyPdfError as e:
        raise ExtractionError(f"Could not read PDF: {e}") from e
