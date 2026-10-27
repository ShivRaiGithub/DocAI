import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import models  # noqa: F401  (registers tables on Base)
from app.api.documents import get_storage
from app.config import Settings, get_settings
from app.db import Base, get_db
from app.embeddings import Embedder, get_embedder
from app.main import app
from app.storage import FileStorage
from app.vectorstore import VectorStore, get_vector_store


@pytest.fixture(scope="session")
def embedder():
    # Real model: downloaded once (~90 MB), then cached by Hugging Face.
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def vector_store():
    # Qdrant's in-memory mode stands in for the Docker server.
    return VectorStore(QdrantClient(":memory:"), "test_chunks", 384)


@pytest.fixture
def client(tmp_path, embedder, vector_store):
    # In-memory SQLite stands in for Postgres so tests need no servers.
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
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    yield TestClient(app)  # no `with`, so the startup hook (real Postgres) is skipped
    app.dependency_overrides.clear()
