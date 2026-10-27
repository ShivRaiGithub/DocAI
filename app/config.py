from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, read from environment variables or a .env file."""

    app_name: str = "DocAI"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/docai"

    data_dir: Path = Path("data")
    max_upload_mb: int = 20
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # must match the model above
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    llm_timeout_s: float = 300.0
    rag_top_k: int = 5
    rag_min_score: float = 0.25
    agent_max_rewrites: int = 2
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "chunks"
    chunk_size: int = 800
    chunk_overlap: int = 200
    allowed_extensions: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

    @field_validator("database_url")
    @classmethod
    def use_psycopg_driver(cls, url: str) -> str:
        """Accept the usual postgres:// and postgresql:// forms and pick our driver."""
        for prefix in ("postgres://", "postgresql://"):
            if url.startswith(prefix):
                return "postgresql+psycopg://" + url[len(prefix):]
        return url

    model_config = SettingsConfigDict(env_prefix="DOCAI_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
