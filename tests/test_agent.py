import pytest

from app.llm import LLMError
from app.rag import clean_rewrite
from tests.test_documents import upload

PASTA = b"To make pasta, boil salted water, add the noodles and cook for ten minutes."
CATS = b"Cats sleep most of the day. A kitten likes warm blankets."
VAGUE = "xyzzy plugh frobnicate"  # shares no meaning with any document


@pytest.fixture
def indexed(client):
    for name, data in {"cooking.txt": PASTA, "cats.txt": CATS}.items():
        doc = upload(client, name, data).json()
        client.post(f"/documents/{doc['id']}/index")


def ask_agent(client, question, **extra):
    return client.post("/ask/agent", json={"question": question, "min_score": 0.3, **extra})


def test_good_first_query_needs_no_rewrite(client, llm, indexed):
    llm.reply = "Ten minutes [1]."
    body = ask_agent(client, "How long do I cook spaghetti noodles?").json()
    assert body["answer"] == "Ten minutes [1]."
    assert body["search_queries"] == ["How long do I cook spaghetti noodles?"]
    assert [s.split(":")[0] for s in body["steps"]] == ["retrieve", "generate"]
    assert llm.structured_calls == []


def test_weak_query_is_rewritten_then_succeeds(client, llm, indexed):
    llm.rewrites = ["boil pasta noodles cooking time"]
    llm.reply = "Ten minutes [1]."
    body = ask_agent(client, VAGUE).json()
    assert body["search_queries"] == [VAGUE, "boil pasta noodles cooking time"]
    assert [s.split(":")[0] for s in body["steps"]] == ["retrieve", "rewrite", "retrieve", "generate"]
    assert body["citations"][0]["filename"] == "cooking.txt"
    assert len(llm.structured_calls) == 1 and len(llm.calls) == 1
    # the rewrite prompt shows the model what was already tried
    assert VAGUE in llm.structured_calls[0][1]["content"]


def test_gives_up_after_max_rewrites_without_answering(client, llm, indexed):
    llm.rewrites = ["gibberish one", "gibberish two", "never used"]
    body = ask_agent(client, VAGUE).json()
    assert len(body["search_queries"]) == 3  # original + 2 rewrites (the default maximum)
    assert len(llm.structured_calls) == 2
    assert llm.calls == []  # no relevant chunks, so the answer LLM is never called
    assert "couldn't find" in body["answer"] and body["steps"][-1].startswith("give_up")


def test_stops_early_when_rewrite_repeats_or_is_empty(client, llm, indexed):
    llm.rewrites = [VAGUE]  # repeats the original
    body = ask_agent(client, VAGUE).json()
    assert body["search_queries"] == [VAGUE] and len(llm.structured_calls) == 1
    assert body["steps"][-1].startswith("give_up")

    llm.rewrites = [""]  # model returns an empty query
    assert ask_agent(client, VAGUE).json()["search_queries"] == [VAGUE]

    llm.rewrites = []  # model reply could not be parsed at all
    assert ask_agent(client, VAGUE).json()["search_queries"] == [VAGUE]


def test_llm_failure_is_503(client, llm, indexed):
    llm.error = LLMError("Could not reach Ollama")
    assert ask_agent(client, VAGUE).status_code == 503


def test_clean_rewrite():
    assert clean_rewrite(' "pasta cooking time" ', []) == "pasta cooking time"
    assert clean_rewrite("  ", []) is None
    assert clean_rewrite("Pasta", ["pasta"]) is None
    assert len(clean_rewrite("x" * 500, [])) == 200
