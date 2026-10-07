import asyncio
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine

from app.auth.keys import hash_key
from app.config import Settings
from app.db.models import ApiKey, Base
from app.db.session import make_engine, make_session_factory
from app.main import create_app
from app.providers.base import GenerationParams, ProviderAdapter, ProviderResponse
from app.ratelimit.service import DatabaseRateLimiter

# Deliberately fake credentials, used only inside the isolated test databases.
ADMIN_KEY = "test-only-admin-credential-over-32-characters"
CLIENT_KEY = "lgw_test-only-client-credential-not-a-real-key"
ADMIN_HEADERS = {"Authorization": f"Bearer {ADMIN_KEY}"}


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 10, 7, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class FakeProvider(ProviderAdapter):
    name = "fake"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, GenerationParams]] = []
        self.failure: BaseException | None = None
        self.healthy = True
        self.gate: asyncio.Event | None = None
        self.input_tokens = 12
        self.output_tokens = 3
        self.started = asyncio.Event()

    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        self.calls.append((prompt, model, params))
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        return ProviderResponse(
            text="Paris",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_ms=1.0,
        )

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
        admin_api_key=ADMIN_KEY,
        cache_enabled=False,
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    # Independent connections are essential: a shared in-memory connection hides transaction races.
    directory = Path(os.environ.get("TEST_SQLITE_DIRECTORY", gettempdir()))
    database = directory / f"gateway-test-{uuid4().hex}.sqlite"
    engine = make_engine(f"sqlite+aiosqlite:///{database}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()
        database.unlink(missing_ok=True)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest_asyncio.fixture
async def api_key(engine: AsyncEngine, clock: Clock) -> ApiKey:
    key = ApiKey(
        id=uuid4(),
        name="test",
        key_hash=hash_key(CLIENT_KEY),
        active=True,
        requests_per_minute=60,
        daily_token_budget=100000,
        bucket_tokens=60.0,
        bucket_updated_at=clock(),
    )
    async with make_session_factory(engine)() as session:
        session.add(key)
        await session.commit()
    return key


@pytest.fixture
def app(
    settings: Settings, provider: FakeProvider, engine: AsyncEngine, api_key: ApiKey, clock: Clock
) -> FastAPI:
    app = create_app(settings=settings, provider=provider, engine=engine)
    app.state.rate_limiter = DatabaseRateLimiter(app.state.session_factory, clock=clock)
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # HTTPX ASGITransport does not run startup/shutdown automatically.
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        ) as client:
            yield client
