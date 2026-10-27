from fastapi import APIRouter

from app.embeddings import EmbedderDep, cosine_similarity
from app.schemas import EmbedRequest, EmbedResponse, SimilarityRequest, SimilarityResponse

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.post("", response_model=EmbedResponse)
def embed_texts(body: EmbedRequest, embedder: EmbedderDep) -> EmbedResponse:
    """Return the embedding vector of each text."""
    vectors = embedder.embed(body.texts)
    return EmbedResponse(
        model=embedder.model_name, dimension=len(vectors[0]), embeddings=vectors
    )


@router.post("/similarity", response_model=SimilarityResponse)
def similarity(body: SimilarityRequest, embedder: EmbedderDep) -> SimilarityResponse:
    """Cosine similarity of two texts, from -1 (opposite) to 1 (same meaning)."""
    a, b = embedder.embed([body.a, body.b])
    return SimilarityResponse(model=embedder.model_name, similarity=cosine_similarity(a, b))
