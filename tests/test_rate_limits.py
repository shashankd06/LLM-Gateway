import asyncio
from uuid import UUID

import httpx
from fastapi import FastAPI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.models import ApiKey, DailyUsage, RequestLog
from app.db.session import make_session_factory
from app.main import create_app
from app.providers.base import ProviderTimeout
from app.ratelimit.service import DatabaseRateLimiter
from tests.conftest import ADMIN_HEADERS, CLIENT_KEY, Clock, FakeProvider

CHAT = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}


async def limits(engine: AsyncEngine, key: ApiKey, rpm: int = 60, budget: int = 100000) -> None:
    async with make_session_factory(engine)() as session:
        stored = await session.get(ApiKey, key.id)
        assert stored is not None
        stored.requests_per_minute = rpm
        stored.bucket_tokens = float(rpm)
        stored.daily_token_budget = budget
        await session.commit()


async def test_rate_limit_and_retry_after(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
    provider: FakeProvider,
) -> None:
    await limits(engine, api_key, rpm=2)
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
    rejected = await client.post("/v1/chat/completions", json=CHAT)
    assert rejected.status_code == 429 and rejected.headers["Retry-After"] == "30"
    assert rejected.json()["error"]["code"] == "rate_limit_exceeded"
    assert len(provider.calls) == 2
    clock.advance(30)
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200


async def test_parallel_requests_never_exceed_capacity_across_app_instances(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    settings: Settings,
    clock: Clock,
    provider: FakeProvider,
) -> None:
    await limits(engine, api_key, rpm=5)
    second_provider = FakeProvider()
    second = create_app(settings=settings, engine=engine, provider=second_provider)
    second.state.rate_limiter = DatabaseRateLimiter(second.state.session_factory, clock)
    async with second.router.lifespan_context(second):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=second),
            base_url="http://test",
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        ) as other:
            responses = await asyncio.gather(
                *[
                    (client if i % 2 == 0 else other).post("/v1/chat/completions", json=CHAT)
                    for i in range(40)
                ]
            )
    assert sum(response.status_code == 200 for response in responses) == 5
    assert sum(response.status_code == 429 for response in responses) == 35
    assert len(provider.calls) + len(second_provider.calls) == 5
    assert all(
        int(response.headers["Retry-After"]) > 0
        for response in responses
        if response.status_code == 429
    )
    async with make_session_factory(engine)() as session:
        logs = list((await session.scalars(select(RequestLog))).all())
        stored = await session.get(ApiKey, api_key.id)
        usage = await session.get(DailyUsage, (api_key.id, clock().date()))
        assert len(logs) == 40 and stored is not None and stored.bucket_tokens == 0
        assert usage is not None and usage.tokens_charged == 75


async def test_key_limits_are_isolated(
    client: httpx.AsyncClient, engine: AsyncEngine, api_key: ApiKey
) -> None:
    await limits(engine, api_key, rpm=1)
    created = await client.post(
        "/admin/keys", headers=ADMIN_HEADERS, json={"name": "independent", "requests_per_minute": 1}
    )
    raw = created.json()["api_key"]
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 429
    assert (
        await client.post(
            "/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {raw}"}
        )
    ).status_code == 200


async def test_daily_budget_rejects_before_provider(
    client: httpx.AsyncClient, engine: AsyncEngine, api_key: ApiKey, provider: FakeProvider
) -> None:
    await limits(engine, api_key, budget=1)
    rejected = await client.post("/v1/chat/completions", json=CHAT)
    assert rejected.status_code == 429 and rejected.headers["Retry-After"] == "43200"
    assert rejected.json()["error"]["code"] == "daily_budget_exceeded"
    assert not provider.calls
    async with make_session_factory(engine)() as session:
        key = await session.get(ApiKey, api_key.id)
        assert key is not None and key.bucket_tokens == 60  # Budget rejection does not consume RPM.


async def test_success_refunds_hold_and_failure_retains_it(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
    provider: FakeProvider,
) -> None:
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
    provider.failure = ProviderTimeout("unknown usage")
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 504
    async with make_session_factory(engine)() as session:
        logs = list(
            (
                await session.scalars(
                    select(RequestLog).order_by(RequestLog.created_at, RequestLog.id)
                )
            ).all()
        )
        success = next(log for log in logs if log.status == "success")
        failure = next(log for log in logs if log.status == "provider_timeout")
        usage = await session.get(DailyUsage, (api_key.id, clock().date()))
        assert success.budget_tokens == 15
        assert failure.budget_tokens > 15 and failure.input_tokens == 0
        assert (
            usage is not None
            and usage.tokens_charged == success.budget_tokens + failure.budget_tokens
        )


async def test_concurrent_budget_holds_block_inflight_overspending(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
) -> None:
    # One request fits, but two simultaneous holds cannot fit. Completed calls refund unused tokens.
    await limits(engine, api_key, budget=500)
    provider.gate = asyncio.Event()
    tasks = [asyncio.create_task(client.post("/v1/chat/completions", json=CHAT)) for _ in range(12)]
    try:
        completed = 0
        for future in asyncio.as_completed(tasks, timeout=15):
            assert (await future).status_code == 429
            completed += 1
            if completed == 11:
                break
        assert len(provider.calls) == 1
        async with make_session_factory(engine)() as session:
            usage = (await session.scalars(select(DailyUsage))).one()
            assert 0 < usage.tokens_charged <= 500
        provider.gate.set()
        responses = await asyncio.gather(*tasks)
        assert sum(response.status_code == 200 for response in responses) == 1
        assert sum(response.status_code == 429 for response in responses) == 11
    finally:
        provider.gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_late_completion_settles_original_day(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
    provider: FakeProvider,
) -> None:
    provider.gate = asyncio.Event()
    yesterday = clock().date()
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
    try:
        async with asyncio.timeout(10):
            await provider.started.wait()
        clock.advance(86400)
        provider.gate.set()
        assert (await task).status_code == 200
        assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 200
        async with make_session_factory(engine)() as session:
            previous = await session.get(DailyUsage, (api_key.id, yesterday))
            current = await session.get(DailyUsage, (api_key.id, clock().date()))
            assert previous is not None and current is not None
            assert previous.tokens_charged == current.tokens_charged == 15
    finally:
        provider.gate.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_settlement_is_idempotent(
    client: httpx.AsyncClient, app: FastAPI, engine: AsyncEngine, api_key: ApiKey, clock: Clock
) -> None:
    response = await client.post("/v1/chat/completions", json=CHAT)
    async with make_session_factory(engine)() as session:
        log = await session.get(RequestLog, UUID(response.headers["X-Request-ID"]))
        assert log is not None
    assert await app.state.rate_limiter.settle(log, usage_known=True)
    async with make_session_factory(engine)() as session:
        usage = await session.get(DailyUsage, (api_key.id, clock().date()))
        assert usage is not None and usage.tokens_charged == 15


async def test_default_output_limit_and_rejection(
    client: httpx.AsyncClient, settings: Settings, provider: FakeProvider
) -> None:
    assert (
        await client.post("/v1/chat/completions", json={"messages": CHAT["messages"]})
    ).status_code == 200
    assert provider.calls[0][2].max_tokens == settings.default_max_output_tokens
    too_large = await client.post(
        "/v1/chat/completions", json={**CHAT, "max_tokens": settings.max_output_tokens + 1}
    )
    assert too_large.status_code == 400 and len(provider.calls) == 1


async def test_provider_budget_overrun_is_accounted_and_blocks_more_calls(
    client: httpx.AsyncClient, engine: AsyncEngine, api_key: ApiKey, provider: FakeProvider
) -> None:
    await limits(engine, api_key, budget=500)
    provider.input_tokens = 1000  # Simulates an unexpected tokenizer/template expansion.
    response = await client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 429 and int(response.headers["Retry-After"]) > 0
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.status == "budget_overrun" and log.budget_tokens == 1003
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 429
    assert len(provider.calls) == 1
