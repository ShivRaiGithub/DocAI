from fastapi import APIRouter

from app.embeddings import EmbedderDep
from app.schemas import SearchRequest, SearchResponse
from app.vectorstore import VectorStoreDep

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search(body: SearchRequest, embedder: EmbedderDep, vectors: VectorStoreDep) -> SearchResponse:
    """Find the chunks whose meaning is closest to the query."""
    [query_vector] = embedder.embed([body.query])
    hits = vectors.search(query_vector, body.limit, body.document_ids, body.min_score)
    return SearchResponse(query=body.query, results=hits)
