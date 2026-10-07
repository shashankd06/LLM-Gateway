import asyncio
from collections.abc import AsyncIterator
from time import perf_counter
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.api.routes import chat_completions
from app.api.schemas import ChatCompletionRequest
from app.config import Settings
from app.db.models import ApiKey, CacheEntry, DailyUsage, RequestLog
from app.db.session import make_session_factory
from app.main import create_app
from app.providers.base import ProviderError, ProviderTimeout
from app.providers.ollama import OllamaAdapter
from app.providers.openai_compat import OpenAICompatibleAdapter
from app.ratelimit.service import DatabaseRateLimiter
from tests.conftest import CLIENT_KEY, Clock
from tests.test_cache import CHAT, NEAR, FakeEmbedder
from tests.test_rate_limits import limits
from tests.test_routing import ScriptedProvider, routing_config


@pytest.fixture
def cheap() -> ScriptedProvider:
    return ScriptedProvider()


@pytest.fixture
def strong() -> ScriptedProvider:
    return ScriptedProvider("openai_compat")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def app(
    settings: Settings,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
    engine: AsyncEngine,
    api_key: ApiKey,
    clock: Clock,
    embedder: FakeEmbedder,
) -> FastAPI:
    config = settings.model_copy(update={"routing_enabled": True})
    app = create_app(
        config,
        provider=cheap,
        engine=engine,
        embedder=embedder,
        providers={"ollama": cheap, "openai_compat": strong},
        routing_config=routing_config(),
    )
    app.state.rate_limiter = DatabaseRateLimiter(app.state.session_factory, clock)
    return app


@pytest_asyncio.fixture
async def client(app: FastAPI, clock: Clock) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        app.state.cache.clock = clock
        app.state.model_router.sleep = AsyncMock()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {CLIENT_KEY}"},
        ) as client:
            yield client


async def test_default_cheap_response_and_provenance(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
) -> None:
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 200 and reply.json()["model"] == "small"
    assert reply.headers["X-Tier"] == "cheap" and reply.headers["X-Provider-Attempts"] == "1"
    assert "X-Escalation-Reason" not in reply.headers and len(cheap.calls) == 1 and not strong.calls
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert (log.model, log.provider, log.tier, log.attempts) == ("small", "ollama", "cheap", 1)
        assert log.budget_tokens == 15 and log.escalation_reason is None


async def test_escalation_counts_both_calls_and_logs_actual_serving_model(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    cheap: ScriptedProvider,
) -> None:
    cheap.outcomes["small"] = ["Paris"]
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 200 and reply.json()["model"] == "large"
    assert (
        reply.headers["X-Tier"] == "strong"
        and reply.headers["X-Escalation-Reason"] == "short_response"
    )
    assert reply.json()["usage"]["total_tokens"] == 30
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.tier == "strong" and log.provider == "openai_compat" and log.attempts == 2
        assert log.escalation_reason == "short_response" and log.budget_tokens == 30
        assert float(log.estimated_cost) == pytest.approx(0.000072)


async def test_failed_cheap_attempts_keep_only_used_unknown_allowances(
    client: httpx.AsyncClient,
    cheap: ScriptedProvider,
    engine: AsyncEngine,
) -> None:
    cheap.outcomes["small"] = [ProviderError("private"), ProviderTimeout("private")]
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 200 and reply.headers["X-Provider-Attempts"] == "3"
    assert reply.headers["X-Escalation-Reason"] == "provider_timeout"
    assert reply.json()["usage"]["total_tokens"] == 15
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        prompt = cheap.calls[0][0]
        allowance = len(prompt.encode()) + 256 + 512
        assert log.budget_tokens == 2 * allowance + 15
        usage = (await session.scalars(select(DailyUsage))).one()
        assert usage.tokens_charged == log.budget_tokens


async def test_strong_alias_and_explicit_model_requests(
    client: httpx.AsyncClient,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
) -> None:
    strong_reply = await client.post("/v1/chat/completions", json=CHAT | {"model": "strong"})
    assert strong_reply.json()["model"] == "large" and not cheap.calls
    assert strong_reply.headers["X-Tier"] == "strong"
    direct = await client.post("/v1/chat/completions", json=CHAT | {"model": "custom-model"})
    assert direct.json()["model"] == "custom-model" and direct.headers["X-Tier"] == "direct"
    assert len(strong.calls) == len(cheap.calls) == 1


async def test_fallback_cache_preserves_model_tier_and_exact_fast_path(
    client: httpx.AsyncClient,
    app: FastAPI,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
    embedder: FakeEmbedder,
    engine: AsyncEngine,
) -> None:
    app.state.settings.cache_enabled = True
    cheap.outcomes["small"] = ["Paris"]
    generated = await client.post("/v1/chat/completions", json=CHAT)
    cached = await client.post("/v1/chat/completions", json=CHAT)
    assert generated.headers["X-Cache"] == "MISS" and cached.headers["X-Cache"] == "HIT"
    assert cached.json()["model"] == "large" and cached.headers["X-Tier"] == "strong"
    assert (
        cached.headers["X-Provider-Attempts"] == "0" and "X-Escalation-Reason" not in cached.headers
    )
    assert cached.json()["usage"]["total_tokens"] == 0
    assert len(cheap.calls) == len(strong.calls) == len(embedder.calls) == 1
    near = await client.post("/v1/chat/completions", json=NEAR)
    assert near.headers["X-Cache"] == "HIT" and near.json()["model"] == "large"
    async with make_session_factory(engine)() as session:
        entry = (await session.scalars(select(CacheEntry))).one()
        assert (entry.model, entry.provider, entry.tier) == ("large", "openai_compat", "strong")
        logs = (await session.scalars(select(RequestLog))).all()
        assert sum(log.attempts for log in logs) == 2
        assert sum(log.budget_tokens for log in logs) == 30


async def test_cheap_exact_hit_skips_embedding_even_when_strong_is_checked_first(
    client: httpx.AsyncClient,
    app: FastAPI,
    embedder: FakeEmbedder,
    strong: ScriptedProvider,
) -> None:
    app.state.settings.cache_enabled = True
    await client.post("/v1/chat/completions", json=CHAT)
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "HIT"
    assert len(embedder.calls) == 1 and not strong.calls


async def test_auto_cache_never_overrides_strong_only_request(
    client: httpx.AsyncClient,
    app: FastAPI,
    strong: ScriptedProvider,
) -> None:
    app.state.settings.cache_enabled = True
    await client.post("/v1/chat/completions", json=CHAT)
    reply = await client.post("/v1/chat/completions", json=CHAT | {"model": "strong"})
    assert reply.headers["X-Cache"] == "MISS" and len(strong.calls) == 1
    assert reply.json()["model"] == "large"


async def test_routing_policy_change_invalidates_previous_cache(
    client: httpx.AsyncClient,
    app: FastAPI,
    cheap: ScriptedProvider,
) -> None:
    app.state.settings.cache_enabled = True
    await client.post("/v1/chat/completions", json=CHAT)
    config = app.state.model_router.config.model_dump()
    config["confidence"]["min_response_chars"] = 13
    app.state.model_router.config = type(app.state.model_router.config).model_validate(config)
    assert (await client.post("/v1/chat/completions", json=CHAT)).headers["X-Cache"] == "MISS"
    assert len(cheap.calls) == 2


async def test_open_circuits_return_503_without_charge_or_calls(
    client: httpx.AsyncClient,
    app: FastAPI,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
    engine: AsyncEngine,
) -> None:
    for breaker in app.state.model_router.breakers.values():
        breaker.failure_threshold = 1
        await breaker.failure(await breaker.acquire())
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 503 and int(reply.headers["Retry-After"]) > 0
    assert reply.json()["error"]["code"] == "provider_circuit_open"
    assert not cheap.calls and not strong.calls
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.attempts == log.budget_tokens == log.input_tokens == log.output_tokens == 0


async def test_cache_hit_remains_available_with_open_provider_circuit(
    client: httpx.AsyncClient,
    app: FastAPI,
) -> None:
    app.state.settings.cache_enabled = True
    await client.post("/v1/chat/completions", json=CHAT)
    for breaker in app.state.model_router.breakers.values():
        breaker.failure_threshold = 1
        await breaker.failure(await breaker.acquire())
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 200 and reply.headers["X-Cache"] == "HIT"


async def test_all_upstream_failures_log_bounded_attempts_and_sanitized_error(
    client: httpx.AsyncClient,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
    engine: AsyncEngine,
) -> None:
    cheap.outcomes["small"] = [ProviderError("private")] * 2
    strong.outcomes["large"] = [ProviderTimeout("private")] * 2
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 504 and "private" not in reply.text
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.attempts == 4 and log.tier == "strong" and log.provider == "openai_compat"
        assert log.budget_tokens > 0 and log.status == "provider_timeout"


async def test_routing_worst_case_hold_rejects_before_any_provider_call(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
) -> None:
    await limits(engine, api_key, budget=1000)
    reply = await client.post("/v1/chat/completions", json=CHAT)
    assert reply.status_code == 429 and not cheap.calls and not strong.calls


async def test_routed_cancelled_call_retains_only_started_attempt_allowance(
    client: httpx.AsyncClient,
    app: FastAPI,
    cheap: ScriptedProvider,
    api_key: ApiKey,
    engine: AsyncEngine,
) -> None:
    cheap.outcomes["small"] = [asyncio.CancelledError()]
    request = Request({"type": "http", "path": "/v1/chat/completions", "headers": []})
    request.state.started = perf_counter()
    with pytest.raises(asyncio.CancelledError):
        await chat_completions(
            ChatCompletionRequest(**CHAT),
            request,
            Response(),
            app.state.settings,
            cheap,
            api_key,
            app.state.rate_limiter,
            app.state.cache,
            app.state.model_router,
        )
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.status == "cancelled" and log.attempts == 1
        assert log.budget_tokens == len(cheap.calls[0][0].encode()) + 256 + 512


async def test_negative_routing_charge_is_rejected(app: FastAPI, api_key: ApiKey) -> None:
    with pytest.raises(ValueError):
        await app.state.rate_limiter.settle(
            RequestLog(api_key_id=api_key.id), usage_known=True, charged_tokens=-1
        )


def test_startup_rejects_disabled_external_routing(settings: Settings, engine: AsyncEngine) -> None:
    config = settings.model_copy(update={"routing_enabled": True})
    with pytest.raises(ValueError, match="OPENAI_COMPAT_ENABLED"):
        create_app(config, engine=engine, routing_config=routing_config())


async def test_parallel_fallbacks_consume_one_rpm_allowance_per_request(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
) -> None:
    await limits(engine, api_key, rpm=3)
    cheap.outcomes["small"] = ["Paris"] * 3
    replies = await asyncio.gather(
        *[client.post("/v1/chat/completions", json=CHAT) for _ in range(12)]
    )
    assert sum(reply.status_code == 200 for reply in replies) == 3
    assert sum(reply.status_code == 429 for reply in replies) == 9
    assert len(cheap.calls) == len(strong.calls) == 3
    async with make_session_factory(engine)() as session:
        usage = (await session.scalars(select(DailyUsage))).one()
        assert usage.tokens_charged == 90


async def test_parallel_requests_cannot_overspend_fallback_reservations(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    api_key: ApiKey,
    cheap: ScriptedProvider,
) -> None:
    await limits(engine, api_key, budget=5000)
    cheap.gate = asyncio.Event()
    tasks = [asyncio.create_task(client.post("/v1/chat/completions", json=CHAT)) for _ in range(12)]
    try:
        async with asyncio.timeout(15):
            await cheap.started.wait()
            completed = asyncio.as_completed(tasks)
            denied = [await next(completed) for _ in range(11)]
        assert all(reply.status_code == 429 for reply in denied)
        assert len(cheap.calls) == 1
        async with make_session_factory(engine)() as session:
            usage = (await session.scalars(select(DailyUsage))).one()
            assert 0 < usage.tokens_charged <= 5000
        cheap.gate.set()
        assert sum(reply.status_code == 200 for reply in await asyncio.gather(*tasks)) == 1
    finally:
        cheap.gate.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("default", ["ollama", "openai_compat"])
async def test_lifespan_builds_both_configured_adapters_without_network_calls(
    settings: Settings,
    engine: AsyncEngine,
    default: str,
) -> None:
    config = settings.model_copy(
        update={
            "routing_enabled": True,
            "openai_compat_enabled": True,
            "default_provider": default,
            "openai_compat_base_url": "http://localhost:9999/v1",
        }
    )
    app = create_app(
        config, engine=engine, routing_config=routing_config(), embedder=FakeEmbedder()
    )
    async with app.router.lifespan_context(app):
        registry = app.state.model_router.providers
        assert isinstance(registry["ollama"], OllamaAdapter)
        assert isinstance(registry["openai_compat"], OpenAICompatibleAdapter)
