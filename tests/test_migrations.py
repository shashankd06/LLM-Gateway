import asyncio
import os
from io import StringIO
from pathlib import Path

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, select, text

from app.config import Settings
from app.db.models import Base, RequestLog
from app.db.session import make_engine, make_session_factory
from app.main import create_app
from app.ratelimit.service import DatabaseRateLimiter
from tests.conftest import ADMIN_HEADERS, ADMIN_KEY, Clock, FakeProvider
from tests.test_cache import CHAT, NEAR, FakeEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_postgres_migration_offline_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "postgresql+asyncpg://test:unused@localhost/test"
    monkeypatch.setenv("DATABASE_URL", url)
    output = StringIO()
    config = Config(str(PROJECT_ROOT / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(PROJECT_ROOT / "app/db/migrations"))
    command.upgrade(config, "head", sql=True)
    ddl = output.getvalue()
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "CREATE TABLE request_logs" in ddl
    assert "estimated_cost NUMERIC(20, 10) NOT NULL" in ddl
    assert "created_at TIMESTAMP WITH TIME ZONE" in ddl
    assert "ix_request_logs_api_key_created_at" in ddl
    output.seek(0)
    output.truncate()
    assert "CREATE TABLE api_keys" in ddl
    assert "CREATE TABLE daily_usage" in ddl
    assert "FOREIGN KEY(api_key_id) REFERENCES api_keys" in ddl
    assert "CREATE TABLE cache_entries" in ddl
    assert "embedding VECTOR(384)" in ddl
    command.downgrade(config, "0003:base", sql=True)
    rollback = output.getvalue()
    assert "DROP TABLE request_logs" in rollback
    assert "DROP EXTENSION" not in rollback
    assert "DROP TABLE api_keys" in rollback
    assert "DROP TABLE cache_entries" in rollback


@pytest.mark.postgres
async def test_real_postgres_migration_and_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only run on an explicitly supplied empty disposable database; it is not erased."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not supplied; PostgreSQL + pgvector integration not run")
    engine = make_engine(url)
    try:
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert not tables, "TEST_DATABASE_URL must point to an empty disposable database"
        monkeypatch.setenv("DATABASE_URL", url)
        config = Config(str(PROJECT_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(PROJECT_ROOT / "app/db/migrations"))
        # The online Alembic environment owns an event loop, so run it in a worker thread.
        await asyncio.to_thread(command.upgrade, config, "head")
        async with engine.connect() as connection:
            actual = await connection.run_sync(
                lambda conn: {c["name"] for c in inspect(conn).get_columns("request_logs")}
            )
            assert actual == set(Base.metadata.tables["request_logs"].columns.keys())
            assert (
                await connection.scalar(
                    text("SELECT extname FROM pg_extension WHERE extname='vector'")
                )
                == "vector"
            )
        app = create_app(
            Settings(
                _env_file=None, database_url=url, admin_api_key=ADMIN_KEY, cache_enabled=False
            ),
            provider=FakeProvider(),
            engine=engine,
        )
        clock = Clock()
        app.state.rate_limiter = DatabaseRateLimiter(app.state.session_factory, clock)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/admin/keys",
                    json={"name": "postgres-test", "requests_per_minute": 5},
                    headers=ADMIN_HEADERS,
                )
                assert response.status_code == 201
                raw_key = response.json()["api_key"]
                response = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {raw_key}"},
                    json={
                        "messages": [{"role": "user", "content": "What is the capital of France?"}]
                    },
                )
                assert response.status_code == 200
                responses = await asyncio.gather(
                    *[
                        client.post(
                            "/v1/chat/completions",
                            json={
                                "messages": [{"role": "user", "content": "Hello"}],
                                "max_tokens": 10,
                            },
                            headers={"Authorization": f"Bearer {raw_key}"},
                        )
                        for _ in range(39)
                    ]
                )
                assert sum(reply.status_code == 200 for reply in responses) == 4
                assert sum(reply.status_code == 429 for reply in responses) == 35
        async with make_session_factory(engine)() as session:
            logs = list((await session.scalars(select(RequestLog))).all())
            assert len(logs) == 40
            assert sum(log.status == "success" for log in logs) == 5
        # Exercise the actual pgvector operator and driver serialization, without Ollama or weights.
        fake = FakeProvider()
        cached_app = create_app(
            Settings(
                _env_file=None, database_url=url, admin_api_key=ADMIN_KEY, cache_ttl_seconds=2
            ),
            provider=fake,
            engine=engine,
            embedder=FakeEmbedder(),
        )
        cached_app.state.rate_limiter = DatabaseRateLimiter(cached_app.state.session_factory, clock)
        async with cached_app.router.lifespan_context(cached_app):
            cached_app.state.cache.clock = clock
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=cached_app), base_url="http://test"
            ) as client:
                created = await client.post(
                    "/admin/keys", headers=ADMIN_HEADERS, json={"name": "pgvector-test"}
                )
                headers = {"Authorization": f"Bearer {created.json()['api_key']}"}
                miss = await client.post("/v1/chat/completions", headers=headers, json=CHAT)
                exact = await client.post("/v1/chat/completions", headers=headers, json=CHAT)
                semantic = await client.post("/v1/chat/completions", headers=headers, json=NEAR)
                assert miss.headers["X-Cache"] == "MISS"
                assert exact.headers["X-Cache"] == semantic.headers["X-Cache"] == "HIT"
                assert len(fake.calls) == 1
                clock.advance(2)
                expired = await client.post("/v1/chat/completions", headers=headers, json=CHAT)
                assert expired.headers["X-Cache"] == "MISS" and len(fake.calls) == 2
    finally:
        await engine.dispose()
