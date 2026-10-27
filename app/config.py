from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration, read from environment variables or a .env file."""

    app_name: str = "DocAI"
    app_version: str = "0.1.0"
    debug: bool = False

    model_config = SettingsConfigDict(env_prefix="DOCAI_", env_file=".env")


@lru_cache
def get_settings() -> Settings:
    return Settings()
