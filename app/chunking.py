from app.schemas import Chunk, ExtractedPage

# Preferred places to cut, best first: paragraph, line, sentence, word.
BREAKS = ("\n\n", "\n", ". ", " ")


def chunk_pages(pages: list[ExtractedPage], size: int, overlap: int) -> list[Chunk]:
    """Split pages into overlapping chunks of at most `size` characters.

    Chunks never cross a page boundary, so every chunk has exactly one page
    number to cite. Offsets (start/end) index into that page's text.
    """
    if size < 50:
        raise ValueError("chunk_size must be at least 50")
    if not 0 <= overlap <= size // 2:
        raise ValueError("chunk_overlap must be between 0 and half of chunk_size")

    chunks: list[Chunk] = []
    for page in pages:
        for start, end in _spans(page.text, size, overlap):
            raw = page.text[start:end]
            text = raw.strip()
            if not text:
                continue
            lead = len(raw) - len(raw.lstrip())
            chunks.append(
                Chunk(
                    index=len(chunks),
                    page=page.number,
                    start=start + lead,
                    end=start + lead + len(text),
                    text=text,
                )
            )
    return chunks


def _spans(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        if end < len(text):
            end = _best_cut(text, pos, end)
        spans.append((pos, end))
        if end >= len(text):
            break
        # step back by `overlap` so neighbouring chunks share some context,
        # but always move forward so the loop terminates
        pos = _word_start(text, max(end - overlap, pos + 1), end)
    return spans


def _best_cut(text: str, pos: int, end: int) -> int:
    """Pick the best break in the back half of the window [pos, end)."""
    floor = pos + (end - pos) // 2  # don't cut so early that chunks get tiny
    for sep in BREAKS:
        i = text.rfind(sep, floor, end)
        if i != -1:
            return i + len(sep)
    return end  # no break found (e.g. one huge word): hard cut


def _word_start(text: str, pos: int, end: int) -> int:
    """Nudge `pos` forward to the start of the next word, so a chunk never begins mid-word."""
    if pos == 0 or text[pos - 1].isspace():
        return pos
    for i in range(pos, end):
        if text[i].isspace():
            return i + 1
    return pos
