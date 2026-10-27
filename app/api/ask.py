from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.embeddings import EmbedderDep
from app.llm import LLMDep, LLMError
from app.schemas import AskRequest, AskResponse
from app.services import answer_question
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
    try:
        return answer_question(
            settings, embedder, vectors, llm,
            body.question, body.limit, body.document_ids, body.min_score,
        )
    except LLMError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
