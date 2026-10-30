import json
import os

import pytest
import requests
from langsmith import Client, tracing_context
from pydantic import SecretStr

from app.agent import run_agent
from app.config import Settings
from app.schemas import Chunk
from app.tracing import configure_tracing


class RecordingSession(requests.Session):
    """Stands in for the network: records what would be sent to LangSmith."""

    def __init__(self):
        super().__init__()
        self.runs = []

    def request(self, method, url, **kwargs):
        if method == "POST" and url.endswith("/runs"):
            self.runs.append(json.loads(kwargs["data"]))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"
        return response


@pytest.fixture
def session():
    return RecordingSession()


@pytest.fixture
def ls_client(session):
    return Client(
        api_url="http://langsmith.test", api_key="test", session=session,
        auto_batch_tracing=False, info={"version": "0.0.0", "batch_ingest_config": {}},
    )


@pytest.fixture
def seeded(embedder, vector_store, llm):
    chunk = Chunk(index=0, page=1, start=0, end=37, text="Items may be returned within 30 days.")
    vector_store.replace_document("d", 1, "policy.md", [chunk], embedder.embed([chunk.text]))
    llm.reply = "30 days [1]"
    return embedder, vector_store, llm


def run(seeded):
    embedder, vectors, llm = seeded
    return run_agent(Settings(), embedder, vectors, llm, "How many days to return?", min_score=0.1)


def test_agent_run_is_traced_as_nested_spans(seeded, ls_client, session):
    with tracing_context(enabled=True, client=ls_client):
        run(seeded)
    ls_client.flush()

    by_name = {r["name"]: r for r in session.runs}
    assert set(by_name) == {"agent", "semantic_search", "generate_answer"}
    root = by_name["agent"]
    assert root.get("parent_run_id") is None
    assert by_name["semantic_search"]["parent_run_id"] == root["id"]
    assert by_name["generate_answer"]["parent_run_id"] == root["id"]
    assert by_name["semantic_search"]["run_type"] == "retriever"


def test_trace_inputs_are_the_meaningful_arguments_only(seeded, ls_client, session):
    with tracing_context(enabled=True, client=ls_client):
        run(seeded)
    ls_client.flush()

    by_name = {r["name"]: r for r in session.runs}
    assert by_name["semantic_search"]["inputs"] == {
        "query": "How many days to return?", "limit": 5, "document_ids": None, "min_score": 0.1,
    }
    sources = by_name["generate_answer"]["inputs"]["sources"]
    assert len(sources) == 1 and "policy.md p1" in sources[0]
    assert set(by_name["agent"]["inputs"]) == {"question", "limit", "document_ids", "min_score"}


def test_nothing_is_sent_when_tracing_is_disabled(seeded, ls_client, session):
    with tracing_context(enabled=False, client=ls_client):
        run(seeded)
    ls_client.flush()
    assert session.runs == []


@pytest.fixture
def clean_env(monkeypatch):
    # setenv-then-delenv registers each name with monkeypatch, so whatever configure_tracing
    # writes is removed again after the test and cannot leak into other tests
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT",
                 "LANGSMITH_HIDE_INPUTS", "LANGSMITH_HIDE_OUTPUTS"):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


def test_configure_tracing_is_off_by_default_and_overrides_stray_env(clean_env, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")  # e.g. left over in the user's shell
    assert configure_tracing(Settings(_env_file=None)) is False
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_configure_tracing_needs_both_the_flag_and_a_key(clean_env):
    assert configure_tracing(Settings(_env_file=None, langsmith_tracing=True)) is False
    assert configure_tracing(Settings(_env_file=None, langsmith_api_key=SecretStr("k"))) is False
    assert "LANGSMITH_API_KEY" not in os.environ


def test_configure_tracing_exports_settings_when_enabled(clean_env):
    settings = Settings(
        _env_file=None,
        langsmith_tracing=True, langsmith_api_key=SecretStr("secret"),
        langsmith_project="my-project", langsmith_hide_content=True,
    )
    assert configure_tracing(settings) is True
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "secret"
    assert os.environ["LANGSMITH_PROJECT"] == "my-project"
    assert os.environ["LANGSMITH_HIDE_INPUTS"] == os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert "secret" not in repr(settings)  # the key is a SecretStr, so it never prints
