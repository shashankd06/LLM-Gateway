from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.api.routes import database_health
from app.config import Settings
from app.main import create_app, validation_error
from app.providers.ollama import OllamaAdapter


async def test_production_lifecycle_owns_engine(settings: Settings) -> None:
    app = create_app(settings=settings)
    async with app.router.lifespan_context(app):
        assert isinstance(app.state.provider, OllamaAdapter)


async def test_database_health_failure() -> None:
    session = AsyncMock(spec=AsyncSession)
    session.execute.side_effect = OperationalError("SELECT 1", {}, Exception("private"))
    assert await database_health(session) is False


async def test_health_database_failure(
    settings: Settings, engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.conftest import FakeProvider

    app = create_app(settings=settings, provider=FakeProvider(), engine=engine)
    monkeypatch.setattr("app.api.routes.database_health", AsyncMock(return_value=False))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")
    assert response.status_code == 503
    assert response.json()["database"] is False


async def test_validation_outside_completion_route() -> None:
    from fastapi.exceptions import RequestValidationError
    from starlette.requests import Request

    request = Request({"type": "http", "path": "/elsewhere", "headers": []})
    response = await validation_error(request, RequestValidationError([]))
    assert response.status_code == 400
