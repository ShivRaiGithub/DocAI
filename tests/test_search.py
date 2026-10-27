import pytest

from tests.test_documents import upload

NOTES = {
    "cats.txt": b"Cats sleep most of the day. A kitten likes warm blankets and chasing string.",
    "finance.txt": b"Quarterly bond yields rose as the central bank raised interest rates.",
    "cooking.txt": b"To make pasta, boil salted water, add the noodles and cook for ten minutes.",
}


@pytest.fixture
def indexed(client):
    ids = {}
    for name, data in NOTES.items():
        doc = upload(client, name, data).json()
        client.post(f"/documents/{doc['id']}/index")
        ids[name] = doc["id"]
    return ids


def search(client, query, **extra):
    r = client.post("/search", json={"query": query, **extra})
    assert r.status_code == 200
    return r.json()["results"]


def test_finds_right_document_by_meaning_not_keywords(client, indexed):
    top = search(client, "How long should I boil spaghetti?")[0]
    assert top["filename"] == "cooking.txt" and top["document_id"] == indexed["cooking.txt"]

    assert search(client, "my pet likes to nap")[0]["filename"] == "cats.txt"
    assert search(client, "central bank monetary policy")[0]["filename"] == "finance.txt"


def test_results_are_ranked_and_carry_citation_info(client, indexed):
    results = search(client, "pasta", limit=3)
    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    top = results[0]
    assert top["page"] == 1 and top["version"] == 1 and top["start"] == 0
    assert NOTES["cooking.txt"].decode()[top["start"] : top["end"]] == top["text"]


def test_limit_document_filter_and_min_score(client, indexed):
    assert len(search(client, "anything", limit=2)) == 2

    only = search(client, "pasta", document_ids=[indexed["cats.txt"]])
    assert {r["filename"] for r in only} == {"cats.txt"}

    assert search(client, "pasta", document_ids=[]) == []
    assert search(client, "pasta", min_score=0.9) == []


def test_empty_index_and_invalid_requests(client):
    assert search(client, "anything at all") == []
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "limit": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "min_score": 2}).status_code == 422
