import asyncio
import os
import sys
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
from tests.conftest import FakeProvider

PROJECT_ROOT = Path(__file__).resolve().parents[1]


async def run_alembic(database_url: str, *args: str) -> str:
    env = {**os.environ, "DATABASE_URL": database_url}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "alembic",
        *args,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    # Avoid returning a live connection string if a driver emits it in a diagnostic.
    assert process.returncode == 0, stderr.decode(errors="replace").replace(
        database_url, "<database>"
    )
    return stdout.decode()


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
    command.downgrade(config, "0001:base", sql=True)
    rollback = output.getvalue()
    assert "DROP TABLE request_logs" in rollback
    assert "DROP EXTENSION" not in rollback


@pytest.mark.postgres
async def test_real_postgres_migration_and_proxy() -> None:
    """Only run on an explicitly supplied empty disposable database; it is not erased."""
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not supplied; PostgreSQL + pgvector integration not run")
    engine = make_engine(url)
    try:
        async with engine.connect() as connection:
            tables = await connection.run_sync(lambda conn: inspect(conn).get_table_names())
        assert not tables, "TEST_DATABASE_URL must point to an empty disposable database"
        await run_alembic(url, "upgrade", "head")
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
            Settings(_env_file=None, database_url=url), provider=FakeProvider(), engine=engine
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "What is the capital of France?"}]
                    },
                )
                assert response.status_code == 200
        async with make_session_factory(engine)() as session:
            log = (await session.scalars(select(RequestLog))).one()
            assert log.status == "success" and log.input_tokens == 12
    finally:
        await engine.dispose()
