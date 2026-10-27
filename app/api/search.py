from fastapi import APIRouter

from app.embeddings import EmbedderDep
from app.schemas import SearchRequest, SearchResponse
from app.services import semantic_search
from app.vectorstore import VectorStoreDep

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, embedder: EmbedderDep, vectors: VectorStoreDep) -> SearchResponse:
    """Find the chunks whose meaning is closest to the query."""
    hits = semantic_search(
        embedder, vectors, body.query, body.limit, body.document_ids, body.min_score
    )
    return SearchResponse(query=body.query, results=hits)
