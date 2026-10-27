import asyncio

import pytest
from mcp import Client
from sqlalchemy.orm import Session

from app.config import Settings
from app.mcp_server import Services, create_server
from app.storage import FileStorage
from tests.test_documents import upload
from tests.test_extraction import make_pdf

PASTA = b"To make pasta, boil salted water, add the noodles and cook for ten minutes."
CATS = b"Cats sleep most of the day. A kitten likes warm blankets."


@pytest.fixture
def mcp_server(client, tmp_path, engine, embedder, vector_store, llm):
    """The MCP server wired to the same fakes as the REST tests (`client` seeds the data)."""
    for name, data in {"cooking.txt": PASTA, "cats.txt": CATS}.items():
        doc = upload(client, name, data).json()
        client.post(f"/documents/{doc['id']}/index")
    pdf = upload(client, "two.pdf", make_pdf("First page text", "Second page text"), "application/pdf")
    svc = Services(
        settings=Settings(data_dir=tmp_path),
        embedder=embedder,
        vectors=vector_store,
        llm=llm,
        storage=FileStorage(tmp_path / "uploads"),
        session_factory=lambda: Session(engine),
    )
    return create_server(svc), pdf.json()["id"]


def call(server, tool, args=None):
    async def run():
        async with Client(server) as c:
            return await c.call_tool(tool, args or {})

    return asyncio.run(run())


def test_server_advertises_four_read_only_tools(mcp_server):
    server, _ = mcp_server

    async def run():
        async with Client(server) as c:
            return (await c.list_tools()).tools

    tools = asyncio.run(run())
    assert {t.name for t in tools} == {
        "search_documents", "ask_documents", "list_documents", "get_document_text"
    }
    assert all(t.annotations.read_only_hint for t in tools)
    assert all(t.description for t in tools)


def test_search_documents_tool(mcp_server):
    server, _ = mcp_server
    result = call(server, "search_documents", {"query": "How long to boil spaghetti?", "limit": 2})
    assert not result.is_error
    hits = result.structured_content["result"]
    assert hits[0]["filename"] == "cooking.txt" and hits[0]["page"] == 1
    assert len(hits) == 2


def test_ask_documents_tool_returns_citations(mcp_server, llm):
    server, _ = mcp_server
    llm.reply = "Ten minutes [1]."
    result = call(server, "ask_documents", {"question": "How long do I cook spaghetti?"})
    assert not result.is_error
    out = result.structured_content
    assert out["answer"] == "Ten minutes [1]."
    assert out["citations"][0]["filename"] == "cooking.txt"


def test_ask_documents_reports_llm_failure_as_tool_error(mcp_server, llm):
    from app.llm import LLMError

    server, _ = mcp_server
    llm.error = LLMError("Could not reach Ollama")
    result = call(server, "ask_documents", {"question": "How long do I cook spaghetti?"})
    assert result.is_error and "Ollama" in result.content[0].text


def test_list_documents_tool(mcp_server):
    server, _ = mcp_server
    docs = call(server, "list_documents").structured_content["result"]
    assert {d["filename"] for d in docs} == {"cooking.txt", "cats.txt", "two.pdf"}


def test_get_document_text_pages_and_truncation(mcp_server):
    server, pdf_id = mcp_server
    full = call(server, "get_document_text", {"document_id": pdf_id}).structured_content
    assert full["page_count"] == 2 and full["pages_returned"] == [1, 2]
    assert "[Page 1]\nFirst page text" in full["text"] and not full["truncated"]

    one = call(server, "get_document_text", {"document_id": pdf_id, "page": 2}).structured_content
    assert one["pages_returned"] == [2] and "First" not in one["text"]

    cut = call(server, "get_document_text", {"document_id": pdf_id, "max_chars": 10}).structured_content
    assert cut["truncated"] and len(cut["text"]) == 10


def test_get_document_text_errors(mcp_server):
    server, pdf_id = mcp_server
    def error_text(args):
        result = call(server, "get_document_text", args)
        assert result.is_error
        return result.content[0].text

    assert "not found" in error_text({"document_id": "nope"})
    assert "has 2 pages" in error_text({"document_id": pdf_id, "page": 9})
    assert "Version 5" in error_text({"document_id": pdf_id, "version": 5})
