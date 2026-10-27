from collections.abc import Iterator
from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from app.config import Settings, get_settings


class Base(DeclarativeBase):
    """Parent of all table models (see app/models.py)."""


@lru_cache
def get_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def get_db(settings: Annotated[Settings, Depends(get_settings)]) -> Iterator[Session]:
    """One database session per request, always closed afterwards."""
    with Session(get_engine(settings.database_url)) as session:
        yield session


def init_db(database_url: str) -> None:
    """Create any missing tables. (A migration tool like Alembic replaces this later.)"""
    from app import models  # noqa: F401  (importing registers the tables on Base)

    Base.metadata.create_all(get_engine(database_url))
