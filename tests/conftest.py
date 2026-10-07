from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.models import Base
from app.db.session import make_engine
from app.main import create_app
from app.providers.base import GenerationParams, ProviderAdapter, ProviderResponse


class FakeProvider(ProviderAdapter):
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, GenerationParams]] = []
        self.failure: Exception | None = None
        self.healthy = True

    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        self.calls.append((prompt, model, params))
        if self.failure is not None:
            raise self.failure
        return ProviderResponse(text="Paris", input_tokens=12, output_tokens=3, latency_ms=1.0)

    async def health(self) -> bool:
        return self.healthy


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite+aiosqlite:///:memory:",
        default_model="test-model",
        input_cost_per_1k="0.002",
        output_cost_per_1k="0.004",
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def app(settings: Settings, provider: FakeProvider, engine: AsyncEngine) -> FastAPI:
    return create_app(settings=settings, provider=provider, engine=engine)


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # HTTPX ASGITransport does not run startup/shutdown automatically.
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
