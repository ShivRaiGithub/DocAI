from tests.test_documents import upload

TEXT = ("Cats sleep a lot and enjoy warm places. " * 30).encode()


def test_index_status_and_stored_payload(client, vector_store):
    doc = upload(client, data=TEXT).json()
    assert client.get(f"/documents/{doc['id']}/index").json()["indexed"] is False

    r = client.post(f"/documents/{doc['id']}/index")
    assert r.status_code == 200
    out = r.json()
    assert out["chunks_indexed"] > 1 and out["version"] == 1

    status = client.get(f"/documents/{doc['id']}/index").json()
    assert status == {
        "document_id": doc["id"], "indexed": True,
        "chunk_count": out["chunks_indexed"], "version": 1,
    }

    points, _ = vector_store.client.scroll("test_chunks", limit=100, with_vectors=True)
    p = points[0]
    assert len(p.vector) == 384
    assert p.payload["filename"] == "notes.txt" and p.payload["page"] == 1
    assert p.payload["document_id"] == doc["id"] and p.payload["text"]


def test_reindex_is_idempotent_and_new_version_replaces_old(client):
    doc = upload(client, data=TEXT).json()
    n1 = client.post(f"/documents/{doc['id']}/index").json()["chunks_indexed"]
    n2 = client.post(f"/documents/{doc['id']}/index").json()["chunks_indexed"]
    assert n1 == n2 == client.get(f"/documents/{doc['id']}/index").json()["chunk_count"]

    client.post(
        f"/documents/{doc['id']}/versions",
        files={"file": ("notes.txt", b"A short second version.", "text/plain")},
    )
    out = client.post(f"/documents/{doc['id']}/index").json()
    assert out["version"] == 2 and out["chunks_indexed"] == 1
    status = client.get(f"/documents/{doc['id']}/index").json()
    assert status["chunk_count"] == 1 and status["version"] == 2


def test_documents_are_indexed_separately_and_delete_removes_vectors(client):
    a = upload(client, data=TEXT).json()
    b = upload(client, "b.txt", b"Totally different note about bonds.").json()
    client.post(f"/documents/{a['id']}/index")
    client.post(f"/documents/{b['id']}/index")

    client.delete(f"/documents/{a['id']}")
    assert client.get(f"/documents/{b['id']}/index").json()["chunk_count"] == 1


def test_index_errors(client):
    assert client.post("/documents/nope/index").status_code == 404
    bad = upload(client, "x.pdf", b"not a pdf", "application/pdf").json()
    assert client.post(f"/documents/{bad['id']}/index").status_code == 422
    doc = upload(client).json()
    assert client.post(f"/documents/{doc['id']}/index?version=7").status_code == 404
