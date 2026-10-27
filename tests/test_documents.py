import pytest
from fastapi.testclient import TestClient

from app.api.documents import get_storage
from app.config import Settings, get_settings
from app.main import app
from app.storage import FileStorage


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=1)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path / "uploads")
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client, name="notes.txt", data=b"hello", ctype="text/plain"):
    return client.post("/documents", files={"file": (name, data, ctype)})


def test_upload_get_list_delete(client):
    r = upload(client)
    assert r.status_code == 201
    doc = r.json()
    assert doc["filename"] == "notes.txt" and doc["size_bytes"] == 5

    assert client.get(f"/documents/{doc['id']}").json() == doc
    assert [d["id"] for d in client.get("/documents").json()] == [doc["id"]]

    assert client.delete(f"/documents/{doc['id']}").status_code == 204
    assert client.get(f"/documents/{doc['id']}").status_code == 404


def test_rejects_unsupported_extension(client):
    assert upload(client, "evil.exe").status_code == 415


def test_rejects_empty_and_oversized(client):
    assert upload(client, data=b"").status_code == 400
    assert upload(client, data=b"x" * (1024 * 1024 + 1)).status_code == 413


def test_path_traversal_filename_is_neutralised(client, tmp_path):
    doc = upload(client, "../../evil.txt").json()
    assert doc["filename"] == "evil.txt"
    assert not (tmp_path.parent / "evil.txt").exists()


def test_unknown_id_and_traversal_id_404(client):
    assert client.get("/documents/nope").status_code == 404
    assert client.delete("/documents/..%2F..").status_code == 404
