import pytest

from app.llm import LLMError
from app.rag import build_messages, cited_numbers
from app.schemas import SearchHit
from tests.test_documents import upload

PASTA = b"To make pasta, boil salted water, add the noodles and cook for ten minutes."
CATS = b"Cats sleep most of the day. A kitten likes warm blankets."


@pytest.fixture
def indexed(client):
    for name, data in {"cooking.txt": PASTA, "cats.txt": CATS}.items():
        doc = upload(client, name, data).json()
        client.post(f"/documents/{doc['id']}/index")


def ask(client, question="How long do I cook spaghetti?", **extra):
    return client.post("/ask", json={"question": question, **extra})


def test_answer_with_citations_and_grounded_prompt(client, llm, indexed):
    llm.reply = "Cook the noodles for ten minutes [1]."
    r = ask(client)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == llm.reply and body["model"] == "fake-model"

    assert [c["filename"] for c in body["citations"]] == ["cooking.txt"]
    assert body["citations"][0]["number"] == 1 and body["citations"][0]["page"] == 1
    assert len(body["sources"]) >= 1

    system, user = llm.calls[0]
    assert system["role"] == "system" and "ONLY" in system["content"]
    assert "[1] (cooking.txt, page 1)" in user["content"]
    assert PASTA.decode() in user["content"]
    assert "Question: How long do I cook spaghetti?" in user["content"]


def test_citations_ignore_numbers_the_model_invented(client, llm, indexed):
    llm.reply = "Ten minutes [1] or maybe [7]."
    body = ask(client, min_score=0.5).json()
    assert [c["number"] for c in body["citations"]] == [1]


def test_no_relevant_sources_skips_the_llm(client, llm, indexed):
    body = ask(client, min_score=0.99).json()
    assert body["citations"] == [] and body["model"] is None
    assert "couldn't find" in body["answer"]
    assert llm.calls == []


def test_empty_index_skips_the_llm(client, llm):
    assert ask(client).json()["sources"] == []
    assert llm.calls == []


def test_document_filter_limits_sources(client, llm, indexed):
    llm.reply = "x"
    docs = {d["filename"]: d["id"] for d in client.get("/documents").json()}
    body = ask(client, min_score=-1, document_ids=[docs["cats.txt"]]).json()
    assert {s["filename"] for s in body["sources"]} == {"cats.txt"}


def test_llm_failure_is_503_and_validation_is_422(client, llm, indexed):
    llm.error = LLMError("Could not reach Ollama")
    r = ask(client)
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]
    assert ask(client, question="").status_code == 422


def test_cited_numbers_and_prompt_numbering():
    assert cited_numbers("a [1][3] b [2] c [9] [0]", 3) == {1, 2, 3}
    assert cited_numbers("no citations", 3) == set()
    hit = SearchHit(score=0.5, text="T", document_id="d", filename="f.pdf", version=1,
                    page=4, chunk_index=0, start=0, end=1)
    user = build_messages("Q?", [hit, hit])[1]["content"]
    assert "[1] (f.pdf, page 4)\nT" in user and "[2] (f.pdf, page 4)" in user
