import asyncio
from datetime import datetime
from time import perf_counter
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request, Response
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.api.routes import chat_completions
from app.api.schemas import ChatCompletionRequest
from app.config import Settings
from app.db.models import ApiKey, DailyUsage, RequestLog
from app.db.session import make_session_factory
from app.errors import GatewayError
from app.providers.base import ProviderTimeout
from app.ratelimit.service import DatabaseRateLimiter
from tests.conftest import Clock, FakeProvider
from tests.test_rate_limits import CHAT, limits


async def test_cancelled_provider_keeps_budget_charge_and_logs_outcome(
    app: FastAPI,
    engine: AsyncEngine,
    settings: Settings,
    provider: FakeProvider,
    api_key: ApiKey,
    clock: Clock,
) -> None:
    provider.failure = asyncio.CancelledError()
    request = Request({"type": "http", "path": "/v1/chat/completions", "headers": []})
    request.state.started = perf_counter()
    with pytest.raises(asyncio.CancelledError):
        await chat_completions(
            ChatCompletionRequest(**CHAT),
            request,
            Response(),
            settings,
            provider,
            api_key,
            app.state.rate_limiter,
        )
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        usage = await session.get(DailyUsage, (api_key.id, clock().date()))
        assert log.status == "cancelled" and log.http_status == 499
        assert usage is not None and usage.tokens_charged == log.budget_tokens > 0


async def test_revocation_during_inference_does_not_lose_accounting(
    client: httpx.AsyncClient, engine: AsyncEngine, api_key: ApiKey, provider: FakeProvider
) -> None:
    provider.gate = asyncio.Event()
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
    try:
        async with asyncio.timeout(10):
            await provider.started.wait()
        async with make_session_factory(engine)() as session:
            key = await session.get(ApiKey, api_key.id)
            assert key is not None
            key.active = False
            await session.commit()
        provider.gate.set()
        assert (await task).status_code == 200
        assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 401
        async with make_session_factory(engine)() as session:
            usage = (await session.scalars(select(DailyUsage))).one()
            assert usage.tokens_charged == 15
    finally:
        provider.gate.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_daily_budget_resets_at_utc_midnight(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
    clock: Clock,
) -> None:
    await limits(engine, api_key, budget=500)
    provider.failure = ProviderTimeout("unknown usage")
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 504
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 429
    clock.advance(86400)
    provider.failure = None
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200


async def test_settlement_requires_an_admitted_request(app: FastAPI, api_key: ApiKey) -> None:
    missing = RequestLog(id=uuid4(), api_key_id=api_key.id)
    with pytest.raises(RuntimeError, match="admitted request"):
        await app.state.rate_limiter.settle(missing, usage_known=True)


async def test_limiter_rechecks_key_inside_admission_transaction(
    app: FastAPI, engine: AsyncEngine, api_key: ApiKey
) -> None:
    async with make_session_factory(engine)() as session:
        stored = await session.get(ApiKey, api_key.id)
        assert stored is not None
        stored.active = False
        await session.commit()
    with pytest.raises(GatewayError) as exc:
        await app.state.rate_limiter.admit(RequestLog(id=uuid4(), api_key_id=api_key.id), 100)
    assert exc.value.status_code == 401


async def test_invalid_reservation_is_rejected(app: FastAPI, api_key: ApiKey) -> None:
    with pytest.raises(ValueError):
        await app.state.rate_limiter.admit(RequestLog(id=uuid4(), api_key_id=api_key.id), 0)


async def test_database_time_and_postgres_lock_statement(clock: Clock) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.get_bind = Mock()
    session.get_bind.return_value.dialect.name = "postgresql"
    session.scalar.return_value = clock()
    limiter = DatabaseRateLimiter(Mock())
    assert await limiter.now(session) == clock()
    assert "clock_timestamp()" in str(session.scalar.call_args.args[0])
    key = Mock(spec=ApiKey)
    key.active = True
    session.scalar.return_value = key
    assert await limiter.lock_key(session, uuid4()) is key
    sql = str(session.scalar.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert sql.endswith("FOR UPDATE")


async def test_sqlite_default_clock_is_aware(engine: AsyncEngine) -> None:
    limiter = DatabaseRateLimiter(make_session_factory(engine))
    async with limiter.sessions() as session:
        now = await limiter.now(session)
        assert isinstance(now, datetime) and now.tzinfo is not None
