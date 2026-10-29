from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app.llm import LLMError
from frontend import docai_client
from frontend.docai_client import ApiError, DocAIClient
from tests.test_documents import upload

APP_FILE = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
PASTA = b"To make pasta, boil salted water, add the noodles and cook for ten minutes."
CATS = b"Cats sleep most of the day. A kitten likes warm blankets."


@pytest.fixture
def api(client):
    """A DocAIClient wired straight into the real FastAPI app (with the test fakes behind it)."""
    return DocAIClient("http://testserver", http=client)


@pytest.fixture
def indexed(client):
    for name, data in {"cooking.txt": PASTA, "cats.txt": CATS}.items():
        doc = upload(client, name, data).json()
        client.post(f"/documents/{doc['id']}/index")


# --- the API client ------------------------------------------------------------------


def test_client_covers_the_document_lifecycle(api):
    doc = api.upload("notes.txt", b"hello docai", "text/plain")
    assert doc["filename"] == "notes.txt" and doc["version"] == 1
    assert [d["id"] for d in api.list_documents()] == [doc["id"]]

    assert api.index_status(doc["id"])["indexed"] is False
    assert api.index(doc["id"])["chunks_indexed"] == 1
    assert api.index_status(doc["id"])["chunk_count"] == 1

    assert api.upload_version(doc["id"], "notes.txt", b"second draft", "text/plain")["version"] == 2
    assert api.text(doc["id"])["pages"][0]["text"] == "second draft"

    api.delete(doc["id"])
    assert api.list_documents() == []


def test_client_search_and_ask(api, llm, indexed):
    hits = api.search("How long should I boil spaghetti?", limit=2)
    assert hits[0]["filename"] == "cooking.txt" and len(hits) == 2

    llm.reply = "Ten minutes [1]."
    plain = api.ask("How long do I cook spaghetti?")
    assert plain["answer"] == "Ten minutes [1]." and "steps" not in plain
    agent = api.ask("How long do I cook spaghetti?", agent=True)
    assert agent["steps"][0].startswith("retrieve")

    cats_id = next(d["id"] for d in api.list_documents() if d["filename"] == "cats.txt")
    only_cats = api.ask("pasta", min_score=-1, document_ids=[cats_id])
    assert {s["filename"] for s in only_cats["sources"]} == {"cats.txt"}


@pytest.mark.parametrize(
    "call,message",
    [
        (lambda a: a.delete("nope"), "Document not found"),
        (lambda a: a.upload("evil.exe", b"x"), "Unsupported file type"),
        (lambda a: a.search(""), "at least 1 character"),  # validation errors arrive as a list
    ],
)
def test_client_turns_http_errors_into_readable_messages(api, call, message):
    with pytest.raises(ApiError, match=message):
        call(api)


def test_client_reports_an_unreachable_or_slow_api():
    def refuse(request):
        raise httpx.ConnectError("refused")

    def stall(request):
        raise httpx.ReadTimeout("slow")

    for handler, text in ((refuse, "Cannot reach the DocAI API"), (stall, "timed out")):
        http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
        with pytest.raises(ApiError, match=text):
            DocAIClient("http://x", http=http).health()


def test_client_reports_llm_outage_from_the_server(api, llm, indexed):
    llm.error = LLMError("Could not reach Ollama at http://localhost:11434")
    with pytest.raises(ApiError, match="Could not reach Ollama"):
        api.ask("How long do I cook spaghetti?")


# --- the pages (Streamlit's headless test runner) --------------------------------------


@pytest.fixture
def use_api(monkeypatch, api):
    """Make every page talk to the test API instead of a real server."""
    st.cache_resource.clear()
    monkeypatch.setattr(docai_client, "make_client", lambda url: api)
    yield
    st.cache_resource.clear()


def ask_page():
    from frontend.views import ask

    ask.render()


def search_page():
    from frontend.views import search

    search.render()


def documents_page():
    from frontend.views import documents

    documents.render()


def run(fn):
    return AppTest.from_function(fn, default_timeout=30).run()


def test_whole_app_shows_the_ask_page_and_connection_status(use_api):
    at = AppTest.from_file(str(APP_FILE), default_timeout=30).run()
    assert not at.exception
    assert any("Ask your documents" in h.value for h in at.header)
    assert any("Connected to DocAI" in s.value for s in at.sidebar.success)


def test_ask_page_answers_with_sources_and_agent_toggle(use_api, llm, indexed):
    llm.reply = "Ten minutes [1]."
    at = run(ask_page)
    at.chat_input[0].set_value("How long do I cook spaghetti?").run()

    assert not at.exception and not at.error
    # exactly one bubble per message: a nested bubble once showed up as a doubled avatar
    assert len(at.chat_message) == 2
    assert not any("Ask a question" in i.value for i in at.info)  # greeting gone once chatting
    texts = [m.markdown[0].value for m in at.chat_message]
    assert texts[0] == "How long do I cook spaghetti?" and "Ten minutes" in texts[1]
    labels = [e.label for e in at.expander]
    assert any(label.startswith("Sources (") for label in labels)
    assert "What the agent did" in labels  # agent mode is on by default

    at.toggle[0].set_value(False)
    at.chat_input[0].set_value("And for cats?").run()
    labels = [e.label for e in at.expander]
    assert labels.count("What the agent did") == 1  # only the first answer used the agent
    assert len([l for l in labels if l.startswith("Sources")]) == 2


def test_ask_page_shows_api_errors_instead_of_crashing(use_api, llm, indexed):
    llm.error = LLMError("Could not reach Ollama")
    at = run(ask_page)
    at.chat_input[0].set_value("How long do I cook spaghetti?").run()
    assert not at.exception
    assert any("Could not reach Ollama" in e.value for e in at.error)
    assert at.session_state["messages"] == []  # the unanswered question is not kept


def test_ask_page_without_documents_still_renders(use_api):
    at = run(ask_page)
    assert not at.exception and any("Ask a question" in i.value for i in at.info)


def test_search_page_lists_ranked_results(use_api, indexed):
    at = run(search_page)
    at.text_input[0].set_value("How long should I boil spaghetti?")
    at.button[0].click().run()
    assert not at.exception
    assert any("cooking.txt" in m.value for m in at.markdown)


def test_source_text_is_shown_as_plain_text_not_rendered_markdown(use_api, client):
    doc = upload(client, "readme.md", b"# Big Heading\n\nSome body text about refunds.").json()
    client.post(f"/documents/{doc['id']}/index")
    at = run(search_page)
    at.text_input[0].set_value("refunds")
    at.button[0].click().run()
    assert any("# Big Heading" in t.value for t in at.text)  # shown literally, not as a heading


def test_search_page_requires_a_query(use_api, indexed):
    at = run(search_page)
    at.button[0].click().run()
    assert any("Type something" in w.value for w in at.warning)


def test_documents_page_lists_status_and_deletes_after_confirmation(use_api, client, indexed):
    client.post("/documents", files={"file": ("plain.txt", b"not indexed yet", "text/plain")})
    at = run(documents_page)
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any(l.startswith("cooking.txt") and "indexed (1 chunks)" in l for l in labels)
    assert any(l.startswith("plain.txt") and "not indexed" in l for l in labels)

    plain_id = next(d["id"] for d in client.get("/documents").json() if d["filename"] == "plain.txt")
    at.button(key=f"delete-{plain_id}").click().run()
    assert any("Delete **plain.txt**" in w.value for w in at.warning)
    assert len(client.get("/documents").json()) == 3  # asking is not deleting

    at.button(key=f"yes-{plain_id}").click().run()
    assert {d["filename"] for d in client.get("/documents").json()} == {"cooking.txt", "cats.txt"}


def test_documents_page_can_index_and_show_text(use_api, client):
    doc = upload(client, "plain.txt", b"read me later").json()
    at = run(documents_page)
    at.button(key=f"index-{doc['id']}").click().run()
    assert client.get(f"/documents/{doc['id']}/index").json()["indexed"] is True

    at.toggle(key=f"text-{doc['id']}").set_value(True).run()
    assert any("read me later" in t.value for t in at.text)


def test_helpers_format_sizes_and_dates(monkeypatch):
    from frontend import state

    assert state.human_size(512) == "512 B" and state.human_size(2048) == "2.0 KB"
    assert state.human_size(5 * 1024 * 1024) == "5.0 MB"

    monkeypatch.setattr(state, "_browser_timezone", lambda: None)
    assert state.short_date("2026-10-27T13:05:00Z") == "27 Oct 2026, 13:05"


def test_dates_are_shown_in_the_viewers_timezone(monkeypatch):
    from frontend import state

    monkeypatch.setattr(state, "_browser_timezone", lambda: "Asia/Kolkata")
    # the same instant, written with two different offsets, reads the same for the viewer
    assert state.short_date("2026-10-27T20:01:00+00:00") == "28 Oct 2026, 01:31"
    assert state.short_date("2026-10-28T01:31:00+05:30") == "28 Oct 2026, 01:31"

    monkeypatch.setattr(state, "_browser_timezone", lambda: "Not/AZone")
    assert state.short_date("2026-10-27T20:01:00+00:00") == "27 Oct 2026, 20:01"  # falls back
