from fastapi import FastAPI

from app.api import documents, health
from app.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.app_version)
    app.include_router(health.router)
    app.include_router(documents.router)
    return app


app = create_app()
