from datetime import datetime

from pydantic import BaseModel


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
