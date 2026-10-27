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
