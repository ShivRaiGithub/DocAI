from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, read from environment variables or a .env file."""

    app_name: str = "DocAI"
    app_version: str = "0.1.0"
    debug: bool = False

    data_dir: Path = Path("data")
    max_upload_mb: int = 20
    allowed_extensions: frozenset[str] = frozenset({".pdf", ".txt", ".md"})

    model_config = SettingsConfigDict(env_prefix="DOCAI_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
