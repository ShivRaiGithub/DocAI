"""Logic shared by the REST API and the MCP server.

Functions here take their dependencies as plain arguments and raise plain
Python exceptions (never HTTP errors), so any interface can call them and map
errors in its own way.
"""

from pathlib import Path

from langsmith import traceable
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.embeddings import Embedder
from app.extraction import extract_pages
from app.llm import OllamaClient
from app.models import Document, DocumentVersion
from app.rag import NO_SOURCES_ANSWER, build_messages, cited_numbers, to_citations
from app.schemas import AskResponse, DocumentInfo, ExtractedPage, SearchHit
from app.storage import FileStorage
from app.vectorstore import VectorStore


def to_info(doc: Document) -> DocumentInfo:
    latest = doc.latest
    return DocumentInfo(
        id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        version=latest.version,
        size_bytes=latest.size_bytes,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


def list_documents(db: Session) -> list[DocumentInfo]:
    docs = db.scalars(select(Document).order_by(Document.created_at.desc())).all()
    return [to_info(d) for d in docs]


def find_version(doc: Document, version: int | None) -> DocumentVersion | None:
    """The requested version, the latest when `version` is None, or None if missing."""
    if version is None:
        return doc.latest
    return next((v for v in doc.versions if v.version == version), None)


def read_pages(storage: FileStorage, version: DocumentVersion) -> list[ExtractedPage]:
    """Text of one stored version. Raises ExtractionError if it can't be read."""
    data = storage.read(version.storage_path)
    return extract_pages(data, Path(version.storage_path).suffix)


@traceable(
    name="semantic_search",
    run_type="retriever",
    # log only the meaningful arguments, not the embedder/vector-store objects
    process_inputs=lambda i: {k: i.get(k) for k in ("query", "limit", "document_ids", "min_score")},
)
def semantic_search(
    embedder: Embedder,
    vectors: VectorStore,
    query: str,
    limit: int,
    document_ids: list[str] | None = None,
    min_score: float | None = None,
) -> list[SearchHit]:
    [query_vector] = embedder.embed([query])
    return vectors.search(query_vector, limit, document_ids, min_score)


@traceable(
    name="answer_question",
    process_inputs=lambda i: {k: i.get(k) for k in ("question", "limit", "document_ids", "min_score")},
)
def answer_question(
    settings: Settings,
    embedder: Embedder,
    vectors: VectorStore,
    llm: OllamaClient,
    question: str,
    limit: int | None = None,
    document_ids: list[str] | None = None,
    min_score: float | None = None,
) -> AskResponse:
    """RAG: retrieve relevant chunks, then have the LLM answer from them. May raise LLMError."""
    hits = semantic_search(
        embedder,
        vectors,
        question,
        limit or settings.rag_top_k,
        document_ids,
        min_score if min_score is not None else settings.rag_min_score,
    )
    return generate_answer(llm, question, hits)


@traceable(
    name="generate_answer",
    process_inputs=lambda i: {
        "question": i["question"],
        "sources": [f"{h.filename} p{h.page} ({h.score:.2f})" for h in i["hits"]],
    },
)
def generate_answer(llm: OllamaClient, question: str, hits: list[SearchHit]) -> AskResponse:
    """Have the LLM answer from the given chunks. With no chunks, skip the LLM entirely."""
    if not hits:
        # Nothing relevant: don't call the LLM rather than risk a made-up answer.
        return AskResponse(
            question=question, answer=NO_SOURCES_ANSWER, model=None, citations=[], sources=[]
        )

    answer = llm.chat(build_messages(question, hits))
    sources = to_citations(hits)
    used = cited_numbers(answer, len(sources))
    return AskResponse(
        question=question,
        answer=answer,
        model=llm.model,
        citations=[c for c in sources if c.number in used],
        sources=sources,
    )
