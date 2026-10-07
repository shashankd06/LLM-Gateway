import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.auth.keys import hash_key
from app.config import Settings
from app.db.models import ApiKey, RequestLog
from app.db.session import make_session_factory
from tests.conftest import ADMIN_HEADERS, CLIENT_KEY, FakeProvider

CHAT = {"messages": [{"role": "user", "content": "Hello"}]}


async def test_key_creation_stores_only_hash_and_can_authenticate(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    response = await client.post(
        "/admin/keys",
        headers=ADMIN_HEADERS,
        json={
            "name": "portfolio-app",
            "requests_per_minute": 7,
            "daily_token_budget": 10000,
        },
    )
    assert response.status_code == 201
    body = response.json()
    raw = body["api_key"]
    assert raw.startswith("lgw_") and len(raw) >= 47
    async with make_session_factory(engine)() as session:
        key = (await session.scalars(select(ApiKey).where(ApiKey.name == "portfolio-app"))).one()
        assert key.key_hash == hash_key(raw)
        assert raw not in key.key_hash and "api_key" not in ApiKey.__table__.columns
        assert (key.requests_per_minute, key.daily_token_budget, key.bucket_tokens) == (7, 10000, 7)
    completion = await client.post(
        "/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {raw}"}
    )
    assert completion.status_code == 200
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert str(log.api_key_id) == body["id"]
    another = await client.post("/admin/keys", headers=ADMIN_HEADERS, json={"name": "second"})
    assert another.status_code == 201 and another.json()["api_key"] != raw


@pytest.mark.parametrize("authorization", ["", "Basic secret", "Bearer wrong", "Bearer ünicode"])
async def test_unknown_key_is_rejected_and_logged(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    provider: FakeProvider,
    authorization: str,
) -> None:
    # HTTP header values must be ASCII; the final case uses a UTF-8 byte value deliberately.
    value = authorization.encode("utf-8")
    response = await client.post(
        "/v1/chat/completions", json=CHAT, headers={b"Authorization": value}
    )
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert not provider.calls
    async with make_session_factory(engine)() as session:
        log = (await session.scalars(select(RequestLog))).one()
        assert log.status == "invalid_api_key" and log.api_key_id is None


async def test_revoked_key_cannot_authenticate(
    client: httpx.AsyncClient, engine: AsyncEngine
) -> None:
    async with make_session_factory(engine)() as session:
        key = await session.scalar(select(ApiKey).where(ApiKey.key_hash == hash_key(CLIENT_KEY)))
        assert key is not None
        key.active = False
        await session.commit()
    assert (await client.post("/v1/chat/completions", json=CHAT)).status_code == 401


@pytest.mark.parametrize("authorization", ["", "Bearer wrong", f"Bearer {CLIENT_KEY}"])
async def test_admin_routes_require_admin_key(
    client: httpx.AsyncClient, authorization: str
) -> None:
    headers = {"Authorization": authorization}
    assert (
        await client.post("/admin/keys", json={"name": "test"}, headers=headers)
    ).status_code == 401
    assert (await client.get("/metrics-summary", headers=headers)).status_code == 401


async def test_admin_is_disabled_without_configuration(
    client: httpx.AsyncClient, settings: Settings
) -> None:
    settings.admin_api_key = None
    response = await client.post("/admin/keys", headers=ADMIN_HEADERS, json={"name": "test"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "admin_unavailable"


@pytest.mark.parametrize(
    "fields",
    [
        {"name": ""},
        {"name": "test", "requests_per_minute": 0},
        {"name": "test", "requests_per_minute": True},
        {"name": "test", "daily_token_budget": 0},
        {"name": "test", "daily_token_budget": 10**13},
        {"name": "test", "extra": "ignored?"},
    ],
)
async def test_key_limits_are_validated(
    client: httpx.AsyncClient, fields: dict[str, object]
) -> None:
    assert (await client.post("/admin/keys", headers=ADMIN_HEADERS, json=fields)).status_code == 400
