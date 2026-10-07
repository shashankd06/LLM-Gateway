from collections.abc import AsyncIterator
from typing import Annotated, cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.keys import require_api_key
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.db.models import ApiKey
from app.providers.base import ProviderAdapter
from app.ratelimit.service import DatabaseRateLimiter
from app.routing.router import ModelRouter


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_provider(request: Request) -> ProviderAdapter:
    return cast(ProviderAdapter, request.app.state.provider)


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session


def get_rate_limiter(request: Request) -> DatabaseRateLimiter:
    return cast(DatabaseRateLimiter, request.app.state.rate_limiter)


def get_cache(request: Request) -> SemanticCache:
    return cast(SemanticCache, request.app.state.cache)


def get_router(request: Request) -> ModelRouter:
    return cast(ModelRouter, request.app.state.model_router)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[ProviderAdapter, Depends(get_provider)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
ApiKeyDep = Annotated[ApiKey, Depends(require_api_key)]
RateLimiterDep = Annotated[DatabaseRateLimiter, Depends(get_rate_limiter)]
CacheDep = Annotated[SemanticCache, Depends(get_cache)]
RouterDep = Annotated[ModelRouter, Depends(get_router)]
