from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app_name": "DocAI",
        "version": "0.1.0",
    }


def test_plain_postgres_urls_get_psycopg_driver():
    from app.config import Settings

    for scheme in ("postgres", "postgresql"):
        s = Settings(database_url=f"{scheme}://u:p@localhost:5432/docai")
        assert s.database_url == "postgresql+psycopg://u:p@localhost:5432/docai"
