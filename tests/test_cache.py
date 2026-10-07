import asyncio
import math
from collections.abc import AsyncIterator
from time import perf_counter
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request, Response
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.api.routes import chat_completions
from app.api.schemas import ChatCompletionRequest
from app.cache.semantic_cache import SemanticCache, cosine_similarity
from app.config import Settings
from app.db.models import ApiKey, CacheEntry, DailyUsage, RequestLog
from app.db.session import make_session_factory
from app.main import create_app
from app.providers.base import GenerationParams, ProviderError, ProviderResponse
from app.ratelimit.service import DatabaseRateLimiter
from tests.conftest import ADMIN_HEADERS, CLIENT_KEY, Clock, FakeProvider
from tests.test_rate_limits import limits

CHAT = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}
NEAR = {"messages": [{"role": "user", "content": "Name France's capital city."}]}


def vector(x: float = 1.0, y: float = 0.0) -> list[float]:
    return [x, y, *([0.0] * 382)]


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.vectors: dict[str, list[float] | None] = {}
        self.failure: BaseException | None = None
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()

    async def embed(self, prompt: str) -> list[float] | None:
        self.calls.append(prompt)
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        for phrase, embedding in self.vectors.items():
            if phrase in prompt:
                return embedding
        return vector()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def app(
    settings: Settings,
    provider: FakeProvider,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
    embedder: FakeEmbedder,
) -> FastAPI:
    config = settings.model_copy(update={"cache_enabled": True, "cache_ttl_seconds": 60})
    app = create_app(config, provider=provider, engine=engine, embedder=embedder)
    app.state.rate_limiter = DatabaseRateLimiter(app.state.session_factory, clock)
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI, clock: Clock) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        app.state.cache.clock = clock
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        ) as client:
            yield client


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (vector(), vector(), 1.0),
        (vector(), vector(0, 1), 0.0),
        (vector(), vector(-1), -1.0),
        (vector(3, 4), vector(6, 8), 1.0),
    ],
)
def test_cosine_similarity(left: list[float], right: list[float], expected: float) -> None:
    assert cosine_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    "invalid", [[], [1.0], vector(0), vector(float("nan")), vector(float("inf"))]
)
def test_cosine_rejects_invalid_vectors(invalid: list[float]) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(vector(), invalid)
    with pytest.raises(ValueError):
        cosine_similarity(invalid, vector())


async def test_exact_fast_path_and_zero_inference_accounting(
    client: httpx.AsyncClient,
    provider: FakeProvider,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
) -> None:
    first = await client.post("/v1/chat/completions", json=CHAT)
    second = await client.post("/v1/chat/completions", json=CHAT)
    assert first.headers["X-Cache"] == "MISS" and second.headers["X-Cache"] == "HIT"
    assert second.json()["choices"] == first.json()["choices"]
    assert second.json()["usage"]["total_tokens"] == 0
    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]
    assert len(provider.calls) == len(embedder.calls) == 1
    async with make_session_factory(engine)() as session:
        usage = await session.get(DailyUsage, (api_key.id, clock().date()))
        logs = list(
            (await session.scalars(select(RequestLog).order_by(RequestLog.created_at))).all()
        )
        assert usage is not None and usage.tokens_charged == 15
        assert {log.cache_status for log in logs} == {"miss", "hit"}
        hit = next(log for log in logs if log.cache_status == "hit")
        assert hit.input_tokens == hit.output_tokens == hit.budget_tokens == hit.estimated_cost == 0
    metrics = (await client.get("/metrics-summary", headers=ADMIN_HEADERS)).json()
    assert metrics["input_tokens"] == 12 and metrics["output_tokens"] == 3


async def test_semantic_hit_uses_new_prompt_embedding(
    client: httpx.AsyncClient,
    provider: FakeProvider,
    embedder: FakeEmbedder,
) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    reply = await client.post("/v1/chat/completions", json=NEAR)
    assert reply.headers["X-Cache"] == "HIT"
    assert len(provider.calls) == 1 and len(embedder.calls) == 2


@pytest.mark.parametrize(("similarity", "status"), [(0.919, "MISS"), (0.92, "HIT"), (0.921, "HIT")])
async def test_threshold_boundary(
    client: httpx.AsyncClient,
    embedder: FakeEmbedder,
    similarity: float,
    status: str,
) -> None:
    embedder.vectors["Name France"] = vector(similarity, math.sqrt(1 - similarity**2))
    await client.post("/v1/chat/completions", json=CHAT)
    assert (await client.post("/v1/chat/completions", json=NEAR)).headers["X-Cache"] == status


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other-model"},
        {"temperature": 0.1},
        {"top_p": 0.5},
        {"max_tokens": 10},
        {"seed": 42},
        {"stop": ["stop"]},
        {"response_format": {"type": "json_object"}},
    ],
)
async def test_model_and_every_generation_parameter_is_scoped(
    client: httpx.AsyncClient,
    provider: FakeProvider,
    change: dict[str, object],
) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    assert (await client.post("/v1/chat/completions", json=CHAT | change)).headers[
        "X-Cache"
    ] == "MISS"
    assert len(provider.calls) == 2


async def test_effective_defaults_match_explicit_values(client: httpx.AsyncClient) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    explicit = CHAT | {"model": "test-model", "max_tokens": 512, "temperature": 0.7, "top_p": 1.0}
    assert (await client.post("/v1/chat/completions", json=explicit)).headers["X-Cache"] == "HIT"


async def test_key_isolation_and_opt_in_sharing(client: httpx.AsyncClient, app: FastAPI) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    created = await client.post("/admin/keys", headers=ADMIN_HEADERS, json={"name": "second"})
    headers = {"Authorization": f"Bearer {created.json()['api_key']}"}
    assert (await client.post("/v1/chat/completions", headers=headers, json=NEAR)).headers[
        "X-Cache"
    ] == "MISS"
    app.state.settings.cache_share_across_keys = True
    assert (await client.post("/v1/chat/completions", headers=headers, json=CHAT)).headers[
        "X-Cache"
    ] == "HIT"
    app.state.settings.cache_share_across_keys = False
    # Its own semantic entry remains valid; a different parameter scope still misses.
    assert (await client.post("/v1/chat/completions", json=NEAR | {"seed": 1})).headers[
        "X-Cache"
    ] == "MISS"


@pytest.mark.parametrize("role", ["system", "user", "assistant"])
async def test_conversation_context_must_match(client: httpx.AsyncClient, role: str) -> None:
    body = {"messages": [{"role": role, "content": "Answer in English"}, *CHAT["messages"]]}
    await client.post("/v1/chat/completions", json=body)
    other = {"messages": [{"role": role, "content": "Answer in French"}, *NEAR["messages"]]}
    assert (await client.post("/v1/chat/completions", json=other)).headers["X-Cache"] == "MISS"


async def test_non_user_final_turn_is_exact_only(
    client: httpx.AsyncClient,
    embedder: FakeEmbedder,
) -> None:
    body = {"messages": [{"role": "assistant", "content": "Finish this text"}]}
    assert (await client.post("/v1/chat/completions", json=body)).headers["X-Cache"] == "MISS"
    assert (await client.post("/v1/chat/completions", json=body)).headers["X-Cache"] == "HIT"
    assert not embedder.calls


async def test_ttl_is_not_extended_by_hits_and_expiry_is_exclusive(
    client: httpx.AsyncClient,
    clock: Clock,
    engine: AsyncEngine,
) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    clock.advance(59)
    assert (await client.post("/v1/chat/completions", json=NEAR)).headers["X-Cache"] == "HIT"
    clock.advance(1)
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "MISS"
    async with make_session_factory(engine)() as session:
        assert await session.scalar(select(func.count(CacheEntry.id))) == 1


async def test_bypass_never_reads_or_writes_cache(
    client: httpx.AsyncClient,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
) -> None:
    for _ in range(2):
        assert (await client.post("/v1/chat/completions", json=CHAT | {"cache": False})).headers[
            "X-Cache"
        ] == "BYPASS"
    assert not embedder.calls
    async with make_session_factory(engine)() as session:
        assert await session.scalar(select(func.count(CacheEntry.id))) == 0
    assert (
        await client.post("/v1/chat/completions", json=CHAT | {"cache": "false"})
    ).status_code == 400


async def test_long_prompt_exact_only(
    client: httpx.AsyncClient,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
) -> None:
    embedder.vectors["capital"] = None
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "MISS"
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "HIT"
    assert (await client.post("/v1/chat/completions", json=NEAR)).headers["X-Cache"] == "MISS"
    async with make_session_factory(engine)() as session:
        entries = (await session.scalars(select(CacheEntry))).all()
        assert all(entry.embedding is None for entry in entries)


@pytest.mark.parametrize("failure", ["embedder", "invalid_vector", "lookup", "write"])
async def test_cache_outages_do_not_break_inference(
    client: httpx.AsyncClient,
    app: FastAPI,
    embedder: FakeEmbedder,
    provider: FakeProvider,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    if failure == "embedder":
        embedder.failure = RuntimeError("private diagnostic")
    elif failure == "invalid_vector":
        embedder.vectors["capital"] = vector(0)
    else:
        method = "lookup" if failure == "lookup" else "store"
        monkeypatch.setattr(app.state.cache, method, AsyncMock(side_effect=RuntimeError("private")))
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 200 and len(provider.calls) == 1
    assert reply.headers["X-Cache"] == ("MISS" if failure == "write" else "BYPASS")
    assert "private" not in reply.text


async def test_provider_errors_are_never_cached(
    client: httpx.AsyncClient,
    provider: FakeProvider,
    engine: AsyncEngine,
) -> None:
    provider.failure = ProviderError("bad")
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "MISS"
    provider.failure = None
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "MISS"
    async with make_session_factory(engine)() as session:
        assert await session.scalar(select(func.count(CacheEntry.id))) == 1


async def test_cache_hits_work_at_exhausted_budget_but_still_consume_rpm(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
) -> None:
    await client.post("/v1/chat/completions", json=CHAT)
    await limits(engine, api_key, rpm=5, budget=15)
    replies = await asyncio.gather(
        *[client.post("/v1/chat/completions", json=CHAT) for _ in range(40)]
    )
    assert sum(reply.status_code == 200 for reply in replies) == 5
    assert sum(reply.status_code == 429 for reply in replies) == 35
    assert all(reply.headers["X-Cache"] == "HIT" for reply in replies if reply.status_code == 200)
    assert len(provider.calls) == 1


async def test_budget_denial_on_miss_has_one_finalized_log_and_no_provider_call(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
) -> None:
    await limits(engine, api_key, budget=1)
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 429 and int(reply.headers["Retry-After"]) > 0
    assert not provider.calls
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.status == "daily_budget_exceeded" and log.budget_tokens == 0


async def test_concurrent_cold_misses_upsert_one_entry(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    provider: FakeProvider,
) -> None:
    provider.gate = asyncio.Event()
    replies = [
        asyncio.create_task(client.post("/v1/chat/completions", json=CHAT)) for _ in range(8)
    ]
    try:
        async with asyncio.timeout(15):
            await provider.started.wait()
        provider.gate.set()
        assert all(reply.status_code == 200 for reply in await asyncio.gather(*replies))
        async with make_session_factory(engine)() as session:
            assert await session.scalar(select(func.count(CacheEntry.id))) == 1
    finally:
        provider.gate.set()
        await asyncio.gather(*replies, return_exceptions=True)


async def test_concurrent_misses_reserve_budget_before_inference(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
) -> None:
    await limits(engine, api_key, budget=500)
    provider.gate = asyncio.Event()
    tasks = [
        asyncio.create_task(client.post("/v1/chat/completions", json=CHAT | {"max_tokens": 10}))
        for _ in range(12)
    ]
    try:
        async with asyncio.timeout(15):
            await provider.started.wait()
            replies = asyncio.as_completed(tasks)
            denied = [await next(replies) for _ in range(11)]
        assert all(reply.status_code == 429 for reply in denied)
        assert len(provider.calls) == 1
        async with make_session_factory(engine)() as session:
            usage = (await session.scalars(select(DailyUsage))).one()
            assert 0 < usage.tokens_charged <= 500
        provider.gate.set()
        assert sum(reply.status_code == 200 for reply in await asyncio.gather(*tasks)) == 1
    finally:
        provider.gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_embedding_cancellation_charges_no_inference(
    client: httpx.AsyncClient,
    app: FastAPI,
    api_key: ApiKey,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
) -> None:
    embedder.failure = asyncio.CancelledError()
    request = Request({"type": "http", "path": "/v1/chat/completions", "headers": []})
    request.state.started = perf_counter()
    with pytest.raises(asyncio.CancelledError):
        await chat_completions(
            ChatCompletionRequest(**CHAT),
            request,
            Response(),
            app.state.settings,
            app.state.provider,
            api_key,
            app.state.rate_limiter,
            app.state.cache,
            app.state.model_router,
        )
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.status == "cancelled" and log.budget_tokens == 0


async def test_revocation_during_lookup_rejects_before_inference(
    client: httpx.AsyncClient,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
    api_key: ApiKey,
    provider: FakeProvider,
) -> None:
    embedder.gate = asyncio.Event()
    task = asyncio.create_task(client.post("/v1/chat/completions", json=CHAT))
    try:
        async with asyncio.timeout(10):
            await embedder.started.wait()
        async with make_session_factory(engine)() as session:
            stored = await session.get(ApiKey, api_key.id)
            assert stored is not None
            stored.active = False
            await session.commit()
        embedder.gate.set()
        reply = await task
        assert reply.status_code == 401 and reply.headers["WWW-Authenticate"] == "Bearer"
        assert not provider.calls
    finally:
        embedder.gate.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_reservation_database_outage_never_calls_provider(
    client: httpx.AsyncClient,
    app: FastAPI,
    provider: FakeProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app.state.rate_limiter,
        "reserve",
        AsyncMock(side_effect=OperationalError("quota write", {}, Exception("private"))),
    )
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 503 and not provider.calls and "private" not in reply.text


async def test_postgres_lookup_uses_cosine_operator_and_filtered_nearest_neighbor(
    settings: Settings,
    clock: Clock,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.get_bind = Mock()
    session.get_bind.return_value.dialect.name = "postgresql"
    session.__aenter__.return_value = session
    entry = CacheEntry(response_text="Paris", finish_reason="stop")
    session.scalar.side_effect = [None, entry]
    sessions = Mock(return_value=session)
    cache = SemanticCache(sessions, FakeEmbedder(), settings, clock)
    query = cache.query(uuid4(), "ollama", "test", GenerationParams(), CHAT["messages"])
    result = await cache.lookup(query, "prompt")
    assert result.response is not None and result.response.text == "Paris"
    statement = session.scalar.call_args.args[0]
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "<=>" in sql and "ORDER BY" in sql and "LIMIT" in sql
    assert "api_key_id =" in sql and "params_hash =" in sql and "expires_at >" in sql
    assert any(
        value == pytest.approx(0.08)
        for value in compiled.params.values()
        if isinstance(value, float)
    )


async def test_postgres_upsert_and_empty_response_policy(settings: Settings, clock: Clock) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.get_bind = Mock()
    session.get_bind.return_value.dialect.name = "postgresql"
    session.__aenter__.return_value = session
    transaction = AsyncMock()
    session.begin = Mock(return_value=transaction)
    cache = SemanticCache(Mock(return_value=session), FakeEmbedder(), settings, clock)
    query = cache.query(uuid4(), "openai_compat", "test", GenerationParams(), CHAT["messages"])
    result = ProviderResponse(text=" ", input_tokens=0, output_tokens=0, latency_ms=0.0)
    await cache.store(query, result, vector())
    session.execute.assert_not_called()
    await cache.store(query, result.model_copy(update={"text": "Paris"}), vector())
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT" in sql and "DO UPDATE" in sql


async def test_reservation_requires_pending_zero_hold(
    app: FastAPI,
    api_key: ApiKey,
) -> None:
    limiter = app.state.rate_limiter
    with pytest.raises(ValueError):
        await limiter.reserve(RequestLog(id=uuid4(), api_key_id=api_key.id), 0)
    with pytest.raises(RuntimeError):
        await limiter.reserve(RequestLog(id=uuid4(), api_key_id=api_key.id), 100)


async def test_inference_budget_day_is_when_miss_reserves_not_when_lookup_starts(
    client: httpx.AsyncClient,
    app: FastAPI,
    clock: Clock,
    engine: AsyncEngine,
    api_key: ApiKey,
) -> None:
    log = RequestLog(
        id=uuid4(),
        api_key_id=api_key.id,
        model="test",
        provider="fake",
        prompt_hash="hash",
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        estimated_cost=0,
        cache_status="miss",
        status="success",
        http_status=200,
    )
    await app.state.rate_limiter.admit(log, 0)
    original_day = clock().date()
    clock.advance(86400)
    await app.state.rate_limiter.reserve(log, 100)
    log.status = "success"
    log.input_tokens = 15
    assert await app.state.rate_limiter.settle(log, usage_known=True)
    async with make_session_factory(engine)() as session:
        previous = await session.get(DailyUsage, (api_key.id, original_day))
        current = await session.get(DailyUsage, (api_key.id, clock().date()))
        assert previous is not None and previous.tokens_charged == 0
        assert current is not None and current.tokens_charged == 15
