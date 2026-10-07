from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.responses import Response

from app.api.routes import error_response, prompt_hash, router
from app.config import Settings
from app.db.models import RequestLog
from app.db.session import make_engine, make_session_factory
from app.metrics.tracking import save_log
from app.providers.base import ProviderAdapter
from app.providers.ollama import OllamaAdapter
from app.providers.openai_compat import OpenAICompatibleAdapter


def build_provider(settings: Settings, client: httpx.AsyncClient) -> ProviderAdapter:
    if settings.default_provider == "openai_compat":
        assert settings.openai_compat_base_url is not None
        key = settings.openai_compat_api_key
        return OpenAICompatibleAdapter(
            client,
            settings.openai_compat_base_url,
            api_key=key.get_secret_value() if key else None,
            health_timeout=settings.health_timeout_seconds,
        )
    return OllamaAdapter(client, settings.ollama_base_url, settings.health_timeout_seconds)


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    if request.url.path == "/v1/chat/completions":
        settings: Settings = request.app.state.settings
        body = exc.body if isinstance(exc.body, dict) else {}
        model = body.get("model")
        messages = body.get("messages")
        log = RequestLog(
            id=uuid4(),
            api_key_id=None,
            model=model[:200] if isinstance(model, str) else settings.default_model,
            provider=request.app.state.provider.name,
            prompt_hash=prompt_hash(messages) if isinstance(messages, list) else None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(perf_counter() - request.state.started) * 1000,
            cache_status="bypass",
            estimated_cost=0,
            status="invalid_request",
            http_status=400,
        )
        try:
            async with request.app.state.session_factory() as session:
                await save_log(session, log)
        except SQLAlchemyError:
            return error_response(503, "Request logging is unavailable", "database_unavailable")
        response = error_response(
            400, "Unsupported or invalid request fields", "invalid_request_error"
        )
        response.headers["X-Request-ID"] = str(log.id)
        return response
    return error_response(400, "Invalid request", "invalid_request_error")


async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    return error_response(503, "Request logging is unavailable", "database_unavailable")


def create_app(
    settings: Settings | None = None,
    provider: ProviderAdapter | None = None,
    engine: AsyncEngine | None = None,
) -> FastAPI:
    """Inject infrastructure in tests; production owns it for the lifetime of the process."""
    config = settings or Settings()
    db_engine = (
        engine if engine is not None else make_engine(config.database_url.get_secret_value())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            async with httpx.AsyncClient(timeout=config.provider_timeout_seconds) as client:
                app.state.provider = (
                    provider if provider is not None else build_provider(config, client)
                )
                yield
        finally:
            if engine is None:
                await db_engine.dispose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.session_factory = make_session_factory(db_engine)

    @app.middleware("http")
    async def start_timer(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.started = perf_counter()
        return await call_next(request)

    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(SQLAlchemyError, database_error)
    app.include_router(router)
    return app
