from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str


class DocumentInfo(BaseModel):
    id: str
    filename: str
    content_type: str | None
    version: int
    size_bytes: int
    created_at: datetime
    updated_at: datetime


class ExtractedPage(BaseModel):
    number: int
    text: str


class ExtractedText(BaseModel):
    document_id: str
    version: int
    filename: str
    page_count: int
    char_count: int
    pages: list[ExtractedPage]


class Chunk(BaseModel):
    index: int
    page: int
    start: int
    end: int
    text: str


class ChunkedDocument(BaseModel):
    document_id: str
    version: int
    chunk_size: int
    chunk_overlap: int
    chunk_count: int
    chunks: list[Chunk]


class EmbedRequest(BaseModel):
    texts: list[Annotated[str, StringConstraints(min_length=1)]] = Field(
        min_length=1, max_length=64
    )


class EmbedResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]


class SimilarityRequest(BaseModel):
    a: str = Field(min_length=1)
    b: str = Field(min_length=1)


class SimilarityResponse(BaseModel):
    model: str
    similarity: float


class IndexResult(BaseModel):
    document_id: str
    version: int
    chunks_indexed: int
    collection: str


class IndexStatus(BaseModel):
    document_id: str
    indexed: bool
    chunk_count: int
    version: int | None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None, description="Only search inside these documents"
    )
    min_score: float | None = Field(
        default=None, ge=-1, le=1, description="Drop results scoring below this"
    )


class SearchHit(BaseModel):
    score: float
    text: str
    document_id: str
    filename: str
    version: int
    page: int
    chunk_index: int
    start: int
    end: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    limit: int | None = Field(default=None, ge=1, le=20, description="Chunks to retrieve")
    document_ids: list[str] | None = None
    min_score: float | None = Field(default=None, ge=-1, le=1)


class Citation(BaseModel):
    number: int
    document_id: str
    filename: str
    version: int
    page: int
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str | None
    citations: list[Citation]
    sources: list[Citation]


class DocumentText(BaseModel):
    document_id: str
    filename: str
    version: int
    page_count: int
    pages_returned: list[int]
    truncated: bool
    text: str


class AgentAskResponse(AskResponse):
    search_queries: list[str]
    steps: list[str]
