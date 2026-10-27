import pytest

from app.chunking import chunk_pages
from app.schemas import ExtractedPage

LOREM = " ".join(f"Sentence number {i} is here." for i in range(200))


def pages(*texts):
    return [ExtractedPage(number=i, text=t) for i, t in enumerate(texts, start=1)]


def test_short_text_is_one_chunk():
    chunks = chunk_pages(pages("Just a little text."), 1000, 200)
    assert len(chunks) == 1 and chunks[0].text == "Just a little text."
    assert (chunks[0].page, chunks[0].start, chunks[0].end) == (1, 0, 19)


def test_chunks_respect_size_and_offsets_match_source():
    chunks = chunk_pages(pages(LOREM), 300, 60)
    assert len(chunks) > 5
    for c in chunks:
        assert len(c.text) <= 300
        assert LOREM[c.start : c.end] == c.text


def test_consecutive_chunks_overlap_and_cover_everything():
    chunks = chunk_pages(pages(LOREM), 300, 60)
    for a, b in zip(chunks, chunks[1:]):
        assert b.start < a.end  # overlap
        assert b.start > a.start  # progress
    assert chunks[0].start == 0 and chunks[-1].end == len(LOREM)


def test_cuts_prefer_paragraph_then_sentence_boundaries():
    text = ("A" * 40 + ". ") * 3 + "\n\n" + ("B" * 40 + ". ") * 3
    first = chunk_pages(pages(text), 150, 0)[0]
    assert first.text.endswith("AAAA.")  # cut after a sentence, not mid-word


def test_chunks_never_cross_pages_and_index_is_global():
    chunks = chunk_pages(pages("a" * 100, "b" * 100), 60, 10)
    assert {c.page for c in chunks} == {1, 2}
    assert all(set(c.text) == ({"a"} if c.page == 1 else {"b"}) for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_unbroken_text_is_hard_cut_and_terminates():
    chunks = chunk_pages(pages("x" * 1000), 100, 20)
    assert all(len(c.text) <= 100 for c in chunks)
    assert chunks[-1].end == 1000


@pytest.mark.parametrize("size,overlap", [(10, 0), (100, 51), (100, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_pages(pages("hello"), size, overlap)


def test_chunks_do_not_start_mid_word():
    text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet " * 10
    for c in chunk_pages(pages(text), 80, 25)[1:]:
        assert text[c.start - 1] == " " or text[c.start - 1] == "\n"
