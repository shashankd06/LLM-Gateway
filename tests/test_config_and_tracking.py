from decimal import Decimal

import httpx
import pytest
from pydantic import ValidationError

from app.api.routes import prompt_hash
from app.config import Settings
from app.main import build_provider
from app.metrics.tracking import estimate_cost
from app.providers.ollama import OllamaAdapter
from app.providers.openai_compat import OpenAICompatibleAdapter


def test_default_local_configuration() -> None:
    settings = Settings(
        _env_file=None, database_url="postgresql+asyncpg://user:secret@localhost/db"
    )
    assert settings.default_provider == "ollama"
    assert not settings.openai_compat_enabled
    assert estimate_cost(500, 100, settings) == 0
    assert "secret" not in repr(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_provider": "openai_compat"},
        {"openai_compat_enabled": True},
        {"provider_timeout_seconds": 0},
        {"input_cost_per_1k": -1},
    ],
)
def test_configuration_rejects_invalid_settings(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, database_url="sqlite+aiosqlite://", **overrides)


def test_cost_calculation(settings: Settings) -> None:
    assert estimate_cost(12, 3, settings) == Decimal("0.000036")


def test_structured_prompt_hash() -> None:
    first = [{"role": "user", "content": "hi"}]
    assert prompt_hash(first) == prompt_hash([{"content": "hi", "role": "user"}])
    assert prompt_hash(first) != prompt_hash([{"role": "system", "content": "hi"}])
    assert prompt_hash(first) != prompt_hash([{"role": "user", "content": "Hi"}])
    assert prompt_hash(first) != prompt_hash(first * 2)


async def test_provider_factory(settings: Settings) -> None:
    async with httpx.AsyncClient() as client:
        assert isinstance(build_provider(settings, client), OllamaAdapter)
        for key in (None, "not-a-real-key"):
            configured = Settings(
                _env_file=None,
                database_url="sqlite+aiosqlite://",
                default_provider="openai_compat",
                openai_compat_enabled=True,
                openai_compat_base_url="http://localhost:9999/v1",
                openai_compat_api_key=key,
            )
            provider = build_provider(configured, client)
            assert isinstance(provider, OpenAICompatibleAdapter)
            assert bool(provider.headers) is (key is not None)
