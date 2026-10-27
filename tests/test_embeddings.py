import pytest

from app.embeddings import cosine_similarity


def test_vectors_have_384_dims_and_unit_length(embedder):
    [v] = embedder.embed(["hello world"])
    assert len(v) == embedder.dimension == 384
    assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-4)


def test_similar_meaning_scores_higher_than_unrelated(embedder):
    cat, kitten, bonds = embedder.embed(
        ["The cat sat on the mat.", "A kitten rests on the rug.", "Quarterly bond yields rose."]
    )
    assert cosine_similarity(cat, kitten) > cosine_similarity(cat, bonds) + 0.3


def test_identical_text_has_similarity_one(embedder):
    a, b = embedder.embed(["same text", "same text"])
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-4)


def test_api_endpoints_with_real_model(client):
    r = client.post("/embeddings", json={"texts": ["a", "b"]})
    assert r.status_code == 200
    body = r.json()
    assert body["dimension"] == 384 and len(body["embeddings"]) == 2

    r = client.post("/embeddings/similarity", json={"a": "dog", "b": "puppy"})
    assert r.json()["similarity"] > 0.5

    assert client.post("/embeddings", json={"texts": []}).status_code == 422
    assert client.post("/embeddings", json={"texts": [""]}).status_code == 422
