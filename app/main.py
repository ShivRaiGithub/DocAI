from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import documents, embeddings, health, search
from app.config import get_settings
from app.db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(get_settings().database_url)  # create tables on startup if missing
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(embeddings.router)
    app.include_router(search.router)
    return app


app = create_app()
