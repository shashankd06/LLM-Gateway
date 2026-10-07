from decimal import Decimal
from uuid import UUID

import httpx
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import RequestLog
from app.db.session import make_session_factory
from app.providers.base import ProviderError, ProviderTimeout
from tests.conftest import FakeProvider

CHAT = {"messages": [{"role": "user", "content": "What is the capital of France?"}]}


async def get_logs(engine: AsyncEngine) -> list[RequestLog]:
    async with make_session_factory(engine)() as session:
        return list((await session.scalars(select(RequestLog))).all())


async def test_completion_and_durable_log(
    client: httpx.AsyncClient, engine: AsyncEngine, provider: FakeProvider
) -> None:
    response = await client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert body["choices"] == [
        {"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}
    ]
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    assert response.headers["X-Cache"] == "BYPASS"
    logs = await get_logs(engine)
    assert len(logs) == 1
    log = logs[0]
    assert log.id == UUID(response.headers["X-Request-ID"])
    assert body["id"] == f"chatcmpl-{log.id}"
    assert log.api_key_id is None
    assert (log.model, log.provider, log.status, log.http_status) == (
        "test-model",
        "fake",
        "success",
        200,
    )
    assert (log.input_tokens, log.output_tokens) == (12, 3)
    assert log.estimated_cost == Decimal("0.0000360000")
    assert log.latency_ms >= 0
    assert log.created_at is not None
    assert log.cache_status == "bypass"
    assert log.prompt_hash is not None and len(log.prompt_hash) == 64
    assert "prompt" not in RequestLog.__table__.columns
    assert len(provider.calls) == 1


async def test_history_and_generation_parameters(
    client: httpx.AsyncClient, provider: FakeProvider
) -> None:
    messages = [
        {"role": "system", "content": "Respond in JSON."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ]
    response = await client.post(
        "/v1/chat/completions",
        json={
            "messages": messages,
            "model": "custom-model",
            "temperature": 0,
            "max_tokens": 10,
            "top_p": 0.5,
            "seed": 7,
            "stop": ["END"],
            "response_format": {"type": "json_object"},
        },
    )
    assert response.status_code == 200
    prompt, model, params = provider.calls[0]
    assert (
        prompt == "[system]\nRespond in JSON.\n\n[user]\nHello\n\n[assistant]\nHi"
        "\n\n[user]\nAgain\n\n[assistant]\n"
    )
    assert model == "custom-model"
    assert params.model_dump() == {
        "temperature": 0.0,
        "top_p": 0.5,
        "max_tokens": 10,
        "seed": 7,
        "stop": ["END"],
        "json_mode": True,
    }


@pytest.mark.parametrize(
    "failure,status,outcome",
    [
        (ProviderError("sensitive upstream error"), 502, "provider_error"),
        (ProviderTimeout("sensitive upstream error"), 504, "provider_timeout"),
        (RuntimeError("sensitive internal error"), 500, "internal_error"),
    ],
)
async def test_failure_is_logged_and_sanitized(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    provider: FakeProvider,
    failure: Exception,
    status: int,
    outcome: str,
) -> None:
    provider.failure = failure
    response = await client.post("/v1/chat/completions", json=CHAT)
    assert response.status_code == status
    assert response.headers["X-Cache"] == "BYPASS"
    assert "sensitive" not in response.text
    log = (await get_logs(engine))[0]
    assert log.status == outcome and log.http_status == status
    assert log.estimated_cost == 0
    assert (log.input_tokens, log.output_tokens) == (0, 0)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"messages": []},
        {**CHAT, "stream": True},
        {**CHAT, "n": 2},
        {**CHAT, "temperature": -1},
        {**CHAT, "max_tokens": 0},
        {**CHAT, "unsupported": "secret-value"},
        {"messages": [{"role": "tool", "content": "secret-value"}]},
        {**CHAT, "response_format": {"type": "json_schema"}},
        {**CHAT, "json_mode": True},
    ],
)
async def test_invalid_request_is_logged_without_inference(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    provider: FakeProvider,
    body: dict[str, object],
) -> None:
    response = await client.post("/v1/chat/completions", json=body)
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "secret-value" not in response.text
    assert not provider.calls
    log = (await get_logs(engine))[0]
    assert log.status == "invalid_request" and log.http_status == 400


async def test_invalid_json(client: httpx.AsyncClient, engine: AsyncEngine) -> None:
    response = await client.post(
        "/v1/chat/completions", content="{", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 400
    log = (await get_logs(engine))[0]
    assert log.status == "invalid_request" and log.prompt_hash is None


async def test_health_and_metrics(client: httpx.AsyncClient, provider: FakeProvider) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": True,
        "provider": {"name": "fake", "healthy": True},
    }
    assert (await client.get("/metrics-summary")).json() == {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0,
    }
    await client.post("/v1/chat/completions", json=CHAT)
    assert (await client.get("/metrics-summary")).json() == {
        "requests": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "estimated_cost": 0.000036,
    }
    provider.healthy = False
    assert (await client.get("/health")).status_code == 503


async def test_database_failure_never_returns_unlogged_success(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    async with engine.begin() as connection:
        await connection.execute(text("DROP TABLE request_logs"))
    for body in (CHAT, {}):
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "database_unavailable"
        assert "request_logs" not in response.text


async def test_unimplemented_api_keys_endpoint(client: httpx.AsyncClient) -> None:
    assert (await client.post("/admin/keys")).status_code == 404
