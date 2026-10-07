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

from app.api.keys import router as keys_router
from app.api.routes import error_response, prompt_hash, router
from app.cache.embedder import Embedder, SentenceTransformerEmbedder
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.db.models import RequestLog
from app.db.session import make_engine, make_session_factory
from app.errors import GatewayError
from app.metrics.tracking import save_log
from app.providers.base import ProviderAdapter
from app.providers.ollama import OllamaAdapter
from app.providers.openai_compat import OpenAICompatibleAdapter
from app.ratelimit.service import DatabaseRateLimiter
from app.routing.config import RoutingConfig, load_routing_config
from app.routing.router import ModelRouter


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


async def rejected_completion(
    request: Request, body_value: object, response: JSONResponse, status: str
) -> JSONResponse:
    if request.url.path == "/v1/chat/completions":
        settings: Settings = request.app.state.settings
        body = body_value if isinstance(body_value, dict) else {}
        model = body.get("model")
        messages = body.get("messages")
        log = RequestLog(
            id=uuid4(),
            api_key_id=getattr(request.state, "api_key_id", None),
            model=model[:200] if isinstance(model, str) else settings.default_model,
            provider=request.app.state.provider.name,
            prompt_hash=prompt_hash(messages) if isinstance(messages, list) else None,
            input_tokens=0,
            output_tokens=0,
            latency_ms=(perf_counter() - request.state.started) * 1000,
            cache_status="bypass",
            estimated_cost=0,
            status=status,
            http_status=response.status_code,
        )
        try:
            async with request.app.state.session_factory() as session:
                await save_log(session, log)
        except SQLAlchemyError:
            return error_response(503, "Request logging is unavailable", "database_unavailable")
        response.headers["X-Request-ID"] = str(log.id)
        response.headers["X-Cache"] = "BYPASS"
    return response


async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return await rejected_completion(
        request,
        exc.body,
        error_response(400, "Unsupported or invalid request fields", "invalid_request_error"),
        "invalid_request",
    )


async def gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
    response = error_response(exc.status_code, exc.message, exc.code)
    if exc.retry_after is not None:
        response.headers["Retry-After"] = str(exc.retry_after)
    if exc.status_code == 401:
        response.headers["WWW-Authenticate"] = "Bearer"
    try:
        body = await request.json()
    except ValueError:
        body = None
    return await rejected_completion(request, body, response, exc.code)


async def database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
    return error_response(503, "Request logging is unavailable", "database_unavailable")


def create_app(
    settings: Settings | None = None,
    provider: ProviderAdapter | None = None,
    engine: AsyncEngine | None = None,
    embedder: Embedder | None = None,
    providers: dict[str, ProviderAdapter] | None = None,
    routing_config: RoutingConfig | None = None,
) -> FastAPI:
    """Inject infrastructure in tests; production owns it for the lifetime of the process."""
    config = settings or Settings()
    routing = routing_config or (
        load_routing_config(config.routing_config_path)
        if config.routing_enabled
        else RoutingConfig()
    )
    if (
        config.routing_enabled
        and not config.openai_compat_enabled
        and any(
            tier.provider == "openai_compat" for tier in (routing.tiers.cheap, routing.tiers.strong)
        )
    ):
        if providers is None or "openai_compat" not in providers:
            raise ValueError("Enable OPENAI_COMPAT_ENABLED before routing to openai_compat")
    db_engine = (
        engine if engine is not None else make_engine(config.database_url.get_secret_value())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        encoder = (
            embedder
            if embedder is not None
            else SentenceTransformerEmbedder(config.embedding_cache_dir)
        )
        app.state.cache = SemanticCache(app.state.session_factory, encoder, config)
        try:
            async with httpx.AsyncClient(timeout=config.provider_timeout_seconds) as client:
                app.state.provider = (
                    provider if provider is not None else build_provider(config, client)
                )
                registry = dict(providers or {})
                registry.setdefault(config.default_provider, app.state.provider)
                if config.routing_enabled:
                    required = {routing.tiers.cheap.provider, routing.tiers.strong.provider}
                    for name in required - registry.keys():
                        # Each adapter shares the client's connection pool, not its circuit state.
                        selected = config.model_copy(update={"default_provider": name})
                        registry[name] = build_provider(selected, client)
                app.state.model_router = ModelRouter(
                    routing,
                    registry,
                    config.default_provider,
                    config.default_model,
                    config.routing_enabled,
                    config.provider_timeout_seconds,
                )
                yield
        finally:
            if embedder is None:
                assert isinstance(encoder, SentenceTransformerEmbedder)
                await encoder.close()
            if engine is None:
                await db_engine.dispose()

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = config
    app.state.session_factory = make_session_factory(db_engine)
    app.state.rate_limiter = DatabaseRateLimiter(app.state.session_factory)
    if provider is not None:
        # Direct unit calls need infrastructure without invoking the ASGI lifespan.
        registry = dict(providers or {})
        registry.setdefault(config.default_provider, provider)
        if (
            not config.routing_enabled
            or {routing.tiers.cheap.provider, routing.tiers.strong.provider} <= registry.keys()
        ):
            app.state.model_router = ModelRouter(
                routing,
                registry,
                config.default_provider,
                config.default_model,
                config.routing_enabled,
                config.provider_timeout_seconds,
            )

    @app.middleware("http")
    async def start_timer(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.started = perf_counter()
        return await call_next(request)

    app.add_exception_handler(RequestValidationError, validation_error)
    app.add_exception_handler(SQLAlchemyError, database_error)
    app.add_exception_handler(GatewayError, gateway_error)
    app.include_router(router)
    app.include_router(keys_router)
    return app
