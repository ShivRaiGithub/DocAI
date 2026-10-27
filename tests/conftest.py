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
from app.llm import get_llm
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


class FakeLLM:
    """Records the prompts it was given and returns canned replies."""

    model = "fake-model"

    def __init__(self):
        self.reply = "No reply set."
        self.rewrites = []  # queued rewritten queries, used in order; empty -> invalid reply
        self.error = None
        self.calls = []  # prompts sent to chat()
        self.structured_calls = []  # prompts sent to structured()

    def chat(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply

    def structured(self, messages, schema):
        self.structured_calls.append(messages)
        if self.error:
            raise self.error
        return schema(query=self.rewrites.pop(0)) if self.rewrites else None


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def engine():
    # In-memory SQLite stands in for Postgres so tests need no servers.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(tmp_path, engine, embedder, vector_store, llm):
    def override_db():
        with Session(engine) as session:
            yield session

    settings = Settings(data_dir=tmp_path, max_upload_mb=1)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path / "uploads")
    app.dependency_overrides[get_embedder] = lambda: embedder
    app.dependency_overrides[get_vector_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm
    yield TestClient(app)  # no `with`, so the startup hook (real Postgres) is skipped
    app.dependency_overrides.clear()
