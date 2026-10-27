"""MCP server: exposes DocAI's knowledge base as tools for external AI clients.

Run over stdio (what Claude Code / Claude Desktop launch):
    uv run python -m app.mcp_server
Or over HTTP:
    uv run python -m app.mcp_server --http

Over stdio, stdout carries the protocol itself, so nothing here may print to it.
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from sqlalchemy.orm import Session

from app import services
from app.config import Settings, get_settings
from app.db import get_engine
from app.embeddings import Embedder, _embedder_for
from app.extraction import ExtractionError
from app.llm import LLMError, OllamaClient
from app.llm import _client_for as _llm_for
from app.models import Document
from app.schemas import AskResponse, DocumentInfo, DocumentText, SearchHit
from app.storage import FileStorage
from app.vectorstore import VectorStore
from app.vectorstore import _client_for as _qdrant_for

INSTRUCTIONS = (
    "DocAI is a private knowledge base of the user's documents. "
    "Use search_documents to find relevant passages, ask_documents to get a sourced answer, "
    "list_documents to see what exists, and get_document_text to read a document. "
    "Search results and answers include filename and page so you can cite them."
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)


@dataclass
class Services:
    """Everything the tools need, bundled so tests can swap in fakes."""

    settings: Settings
    embedder: Embedder
    vectors: VectorStore
    llm: OllamaClient
    storage: FileStorage
    session_factory: Callable[[], Session]


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    return Services(
        settings=settings,
        embedder=_embedder_for(settings.embedding_model),
        vectors=VectorStore(
            _qdrant_for(settings.qdrant_url),
            settings.qdrant_collection,
            settings.embedding_dimension,
        ),
        llm=_llm_for(settings.ollama_url, settings.llm_model, settings.llm_timeout_s),
        storage=FileStorage(settings.data_dir / "uploads"),
        session_factory=lambda: Session(get_engine(settings.database_url)),
    )


def create_server(svc: Services) -> MCPServer:
    server = MCPServer("DocAI", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    def search_documents(
        query: str,
        limit: int = 5,
        document_ids: list[str] | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        """Semantic search over the indexed documents.

        Returns the passages whose meaning is closest to `query`, best first, each with
        its score, filename, page and text. `document_ids` restricts the search;
        `min_score` (-1 to 1, around 0.25 is a sensible cutoff) drops weak matches.
        """
        limit = max(1, min(limit, 20))
        return services.semantic_search(
            svc.embedder, svc.vectors, query, limit, document_ids, min_score
        )

    @server.tool(annotations=READ_ONLY)
    def ask_documents(
        question: str,
        limit: int | None = None,
        document_ids: list[str] | None = None,
    ) -> AskResponse:
        """Answer a question from the documents using a local LLM, with numbered citations.

        `answer` contains markers like [1]; `citations` says which filename and page each
        marker refers to. If nothing relevant is found the answer says so.
        """
        try:
            return services.answer_question(
                svc.settings, svc.embedder, svc.vectors, svc.llm,
                question, limit, document_ids,
            )
        except LLMError as e:
            raise ToolError(str(e)) from e

    @server.tool(annotations=READ_ONLY)
    def list_documents() -> list[DocumentInfo]:
        """List every stored document with its id, filename, version and size."""
        with svc.session_factory() as db:
            return services.list_documents(db)

    @server.tool(annotations=READ_ONLY)
    def get_document_text(
        document_id: str,
        version: int | None = None,
        page: int | None = None,
        max_chars: int = 20_000,
    ) -> DocumentText:
        """Read a document's text (latest version unless `version` is given).

        `page` returns a single page. Output is cut at `max_chars` characters
        (`truncated` tells you); request specific pages to read more of a long document.
        """
        with svc.session_factory() as db:
            doc = db.get(Document, document_id)
            if doc is None:
                raise ToolError(f"Document '{document_id}' not found")
            chosen = services.find_version(doc, version)
            if chosen is None:
                raise ToolError(f"Version {version} of '{document_id}' not found")
            filename = doc.filename
        try:
            pages = services.read_pages(svc.storage, chosen)
        except ExtractionError as e:
            raise ToolError(str(e)) from e
        page_count = len(pages)
        if page is not None:
            pages = [p for p in pages if p.number == page]
            if not pages:
                raise ToolError(f"Page {page} not found (document has {page_count} pages)")

        text = "\n\n".join(f"[Page {p.number}]\n{p.text}" for p in pages)
        truncated = len(text) > max_chars
        return DocumentText(
            document_id=document_id,
            filename=filename,
            version=chosen.version,
            page_count=page_count,
            pages_returned=[p.number for p in pages],
            truncated=truncated,
            text=text[:max_chars],
        )

    return server


def main() -> None:
    server = create_server(build_services())
    if "--http" in sys.argv:
        server.run("streamable-http")
    else:
        server.run("stdio")


if __name__ == "__main__":
    main()
