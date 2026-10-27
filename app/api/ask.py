from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.embeddings import EmbedderDep
from app.llm import LLMDep, LLMError
from app.rag import NO_SOURCES_ANSWER, build_messages, cited_numbers, to_citations
from app.schemas import AskRequest, AskResponse
from app.vectorstore import VectorStoreDep

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    settings: Annotated[Settings, Depends(get_settings)],
    embedder: EmbedderDep,
    vectors: VectorStoreDep,
    llm: LLMDep,
) -> AskResponse:
    """Answer a question from the indexed documents, with citations (RAG)."""
    [query_vector] = embedder.embed([body.question])
    hits = vectors.search(
        query_vector,
        body.limit or settings.rag_top_k,
        body.document_ids,
        body.min_score if body.min_score is not None else settings.rag_min_score,
    )
    if not hits:
        # Nothing relevant: skip the LLM entirely rather than risk a made-up answer.
        return AskResponse(
            question=body.question, answer=NO_SOURCES_ANSWER, model=None, citations=[], sources=[]
        )

    try:
        answer = llm.chat(build_messages(body.question, hits))
    except LLMError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e

    sources = to_citations(hits)
    used = cited_numbers(answer, len(sources))
    return AskResponse(
        question=body.question,
        answer=answer,
        model=llm.model,
        citations=[c for c in sources if c.number in used],
        sources=sources,
    )
