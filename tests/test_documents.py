import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401
from app.api.documents import get_storage
from app.config import Settings, get_settings
from app.db import Base, get_db
from app.main import app
from app.storage import FileStorage


@pytest.fixture
def client(tmp_path):
    # In-memory SQLite stands in for Postgres so tests need no server.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)

    def override_db():
        with Session(engine) as session:
            yield session

    settings = Settings(data_dir=tmp_path, max_upload_mb=1)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path / "uploads")
    yield TestClient(app)  # no `with`, so the startup hook (real Postgres) is skipped
    app.dependency_overrides.clear()


def upload(client, name="notes.txt", data=b"hello", ctype="text/plain"):
    return client.post("/documents", files={"file": (name, data, ctype)})


def test_upload_get_list_delete(client, tmp_path):
    r = upload(client)
    assert r.status_code == 201
    doc = r.json()
    assert doc["filename"] == "notes.txt" and doc["size_bytes"] == 5 and doc["version"] == 1
    assert (tmp_path / "uploads" / doc["id"] / "v1.txt").read_bytes() == b"hello"

    assert client.get(f"/documents/{doc['id']}").json() == doc
    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404
    assert not (tmp_path / "uploads" / doc["id"]).exists()


def test_new_version_bumps_version_and_keeps_old_file(client, tmp_path):
    doc = upload(client).json()
    r = client.post(
        f"/documents/{doc['id']}/versions",
        files={"file": ("notes.md", b"updated text", "text/markdown")},
    )
    assert r.status_code == 201
    new = r.json()
    assert new["version"] == 2 and new["size_bytes"] == 12 and new["filename"] == "notes.md"
    assert (tmp_path / "uploads" / doc["id"] / "v1.txt").exists()
    assert (tmp_path / "uploads" / doc["id"] / "v2.md").exists()


def test_new_version_of_unknown_document_404(client):
    r = client.post("/documents/nope/versions", files={"file": ("a.txt", b"x", "text/plain")})
    assert r.status_code == 404


def test_rejects_unsupported_extension(client):
    assert upload(client, "evil.exe").status_code == 415


def test_rejects_empty_and_oversized(client):
    assert upload(client, data=b"").status_code == 400
    assert upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413


def test_path_traversal_filename_is_neutralised(client, tmp_path):
    doc = upload(client, "../../evil.txt").json()
    assert doc["filename"] == "evil.txt"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_unknown_id_404(client):
    assert client.get("/documents/nope").status_code == 404
    assert client.delete("/documents/nope").status_code == 404


def test_text_endpoint_returns_latest_and_specific_version(client):
    doc = upload(client, data=b"first draft").json()
    client.post(
        f"/documents/{doc['id']}/versions",
        files={"file": ("notes.txt", b"second draft", "text/plain")},
    )
    latest = client.get(f"/documents/{doc['id']}/text").json()
    assert latest["version"] == 2 and latest["pages"][0]["text"] == "second draft"
    assert latest["char_count"] == 12 and latest["page_count"] == 1

    v1 = client.get(f"/documents/{doc['id']}/text?version=1").json()
    assert v1["pages"][0]["text"] == "first draft"
    assert client.get(f"/documents/{doc['id']}/text?version=9").status_code == 404


def test_text_endpoint_422_for_unreadable_file(client):
    doc = upload(client, "bad.pdf", b"not really a pdf", "application/pdf").json()
    assert client.get(f"/documents/{doc['id']}/text").status_code == 422


def test_chunks_endpoint(client):
    body = ("word " * 400).encode()
    doc = upload(client, data=body).json()
    r = client.get(f"/documents/{doc['id']}/chunks?chunk_size=200&chunk_overlap=40")
    assert r.status_code == 200
    out = r.json()
    assert out["chunk_count"] == len(out["chunks"]) > 5
    assert out["chunk_size"] == 200 and out["chunk_overlap"] == 40
    assert all(len(c["text"]) <= 200 for c in out["chunks"])


def test_chunks_endpoint_defaults_and_bad_params(client):
    doc = upload(client, data=b"short note").json()
    out = client.get(f"/documents/{doc['id']}/chunks").json()
    assert out["chunk_size"] == 800 and out["chunk_overlap"] == 200 and out["chunk_count"] == 1
    assert client.get(f"/documents/{doc['id']}/chunks?chunk_size=10").status_code == 422
    assert client.get("/documents/nope/chunks").status_code == 404
