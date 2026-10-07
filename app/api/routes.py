import asyncio
import hashlib
import json
import logging
from time import perf_counter, time
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    AssistantMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Usage,
)
from app.auth.keys import require_admin
from app.cache.semantic_cache import CacheLookup
from app.config import Settings
from app.db.models import RequestLog
from app.deps import (
    ApiKeyDep,
    CacheDep,
    ProviderDep,
    RateLimiterDep,
    RouterDep,
    SessionDep,
    SettingsDep,
)
from app.errors import GatewayError
from app.metrics.tracking import estimate_cost
from app.providers.base import ProviderError, ProviderTimeout
from app.ratelimit.token_bucket import until_midnight
from app.routing.circuit_breaker import CircuitOpen
from app.routing.router import ModelRouter, RouteTrace

router = APIRouter()
logger = logging.getLogger(__name__)


def prompt_hash(messages: list[dict[str, str]]) -> str:
    # Hash structured messages so role/content boundaries cannot collide.
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def error_response(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "param": None, "code": code}},
    )


def record_trace(
    log: RequestLog, trace: RouteTrace, router: ModelRouter, settings: Settings
) -> None:
    if trace.target is not None:
        log.model = trace.target.model
        log.provider = router.providers[trace.target.provider].name
        log.tier = trace.target.tier
    log.attempts, log.escalation_reason = trace.attempts, trace.escalation_reason
    log.input_tokens, log.output_tokens = trace.input_tokens, trace.output_tokens
    log.estimated_cost = estimate_cost(trace.input_tokens, trace.output_tokens, settings)


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    response: Response,
    settings: SettingsDep,
    provider: ProviderDep,
    key: ApiKeyDep,
    limiter: RateLimiterDep,
    cache: CacheDep,
    model_router: RouterDep,
) -> ChatCompletionResponse | JSONResponse:
    plan = model_router.plan(body.model)
    first_target = plan.targets[0]
    model = first_target.model
    messages = [message.model_dump() for message in body.messages]
    # The prompt-based adapter contract preserves text history but not native chat-role semantics.
    prompt = "\n\n".join(f"[{message.role}]\n{message.content}" for message in body.messages)
    prompt += "\n\n[assistant]\n"
    params = body.generation_params()
    params.max_tokens = params.max_tokens or settings.default_max_output_tokens
    if params.max_tokens > settings.max_output_tokens:
        raise GatewayError(400, "max_tokens exceeds the gateway limit", "invalid_request_error")
    # Byte count + template padding errs high for common tokenizers; actual usage is reconciled.
    attempt_allowance = (
        len(prompt.encode("utf-8")) + settings.budget_input_token_padding + params.max_tokens
    )
    reserved_tokens = attempt_allowance * plan.max_attempts * len(plan.targets)
    log = RequestLog(
        id=uuid4(),
        api_key_id=key.id,
        model=model,
        provider=model_router.providers[first_target.provider].name,
        tier=first_target.tier,
        attempts=0,
        prompt_hash=prompt_hash(messages),
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        cache_status="bypass",
        estimated_cost=0,
        status="success",
        http_status=200,
    )
    use_cache = settings.cache_enabled and body.cache
    # Debit RPM before expensive embedding work. Hits need no inference-budget hold.
    await limiter.admit(log, 0 if use_cache else reserved_tokens)
    log.status = "success"
    failure: JSONResponse | None = None
    result = None
    lookup = CacheLookup(None, None)
    queries = []
    trace = RouteTrace()
    try:
        if use_cache:
            queries = [
                cache.query(
                    key.id,
                    model_router.providers[target.provider].name,
                    target.model,
                    params,
                    messages,
                    plan.policy,
                )
                for target in reversed(plan.targets)
            ]
            try:
                lookup = (
                    await cache.lookup(queries[0], prompt)
                    if len(queries) == 1
                    else await cache.lookup_many(queries, prompt)
                )
                log.cache_status = "hit" if lookup.response is not None else "miss"
            except Exception:
                # Cache outages degrade to inference; audit and quota storage still fail closed.
                logger.warning("Cache lookup unavailable; bypassing cache")
                use_cache = False
            result = lookup.response
            if result is not None:
                log.model, log.provider = lookup.model or model, lookup.provider or log.provider
                log.tier = lookup.tier or first_target.tier
            if result is None:
                await limiter.reserve(log, reserved_tokens)
        if result is None:
            result = await model_router.generate(plan, prompt, params, trace, attempt_allowance)
    except asyncio.CancelledError:
        log.status, log.http_status = "cancelled", 499
        record_trace(log, trace, model_router, settings)
        log.latency_ms = (perf_counter() - request.state.started) * 1000
        await asyncio.shield(
            limiter.settle(log, usage_known=True, charged_tokens=trace.budget_charge)
        )
        raise
    except GatewayError as exc:
        # Budget denial after a miss must finalize the existing log, not create a second row.
        log.status, log.http_status = exc.code, exc.status_code
        failure = error_response(exc.status_code, exc.message, exc.code)
        if exc.retry_after is not None:
            failure.headers["Retry-After"] = str(exc.retry_after)
        if exc.status_code == 401:
            failure.headers["WWW-Authenticate"] = "Bearer"
    except SQLAlchemyError:
        # Quota/audit failures must never allow an unaccounted provider call.
        raise
    except CircuitOpen as exc:
        log.status, log.http_status = "provider_circuit_open", 503
        failure = error_response(503, "Provider circuit is open", "provider_circuit_open")
        failure.headers["Retry-After"] = str(exc.retry_after)
    except ProviderTimeout:
        log.status, log.http_status = "provider_timeout", 504
        failure = error_response(504, "Provider timed out", "provider_timeout")
    except ProviderError:
        log.status, log.http_status = "provider_error", 502
        failure = error_response(
            502, "Provider is unavailable or returned invalid data", "provider_error"
        )
    except Exception:
        # Persist unexpected adapter failures too, without exposing exception text to clients.
        log.status, log.http_status = "internal_error", 500
        failure = error_response(500, "An internal provider error occurred", "internal_error")

    # Gateway latency includes preparation + upstream I/O, measured before committing the audit row.
    record_trace(log, trace, model_router, settings)
    log.latency_ms = (perf_counter() - request.state.started) * 1000
    within_budget = await limiter.settle(log, usage_known=True, charged_tokens=trace.budget_charge)
    if not within_budget:
        failure = error_response(
            429, "Provider usage exceeded the budget reservation", "daily_budget_exceeded"
        )
        # The relevant hold belongs to its original UTC day. Retry uses the next UTC reset.
        async with limiter.sessions() as session:
            failure.headers["Retry-After"] = str(until_midnight(await limiter.now(session)))
    if failure is not None:
        failure.headers["X-Request-ID"] = str(log.id)
        failure.headers["X-Cache"] = log.cache_status.upper()
        return failure
    assert result is not None
    if use_cache and log.cache_status == "miss":
        query = next(
            query
            for query in queries
            if query.model == log.model and query.provider == log.provider
        )
        try:
            await cache.store(query, result, lookup.embedding, log.tier or "direct")
        except Exception:
            logger.warning("Cache write unavailable; response was served without storage")
    response.headers["X-Request-ID"] = str(log.id)
    response.headers["X-Cache"] = log.cache_status.upper()
    response.headers["X-Tier"] = log.tier or "direct"
    response.headers["X-Provider-Attempts"] = str(log.attempts)
    if log.escalation_reason is not None:
        response.headers["X-Escalation-Reason"] = log.escalation_reason
    return ChatCompletionResponse(
        id=f"chatcmpl-{log.id}",
        created=int(time()),
        model=log.model,
        choices=[
            Choice(
                message=AssistantMessage(content=result.text), finish_reason=result.finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=log.input_tokens,
            completion_tokens=log.output_tokens,
            total_tokens=log.input_tokens + log.output_tokens,
        ),
    )


async def database_health(session: AsyncSession) -> bool:
    try:
        await session.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError:
        return False


@router.get("/health")
async def health(provider: ProviderDep, session: SessionDep) -> JSONResponse:
    database_ok, provider_ok = await asyncio.gather(database_health(session), provider.health())
    healthy = database_ok and provider_ok
    return JSONResponse(
        status_code=200 if healthy else 503,
        content={
            "status": "ok" if healthy else "degraded",
            "database": database_ok,
            "provider": {"name": provider.name, "healthy": provider_ok},
        },
    )


@router.get("/metrics-summary", dependencies=[Depends(require_admin)])
async def metrics_summary(session: SessionDep) -> dict[str, int | float]:
    row = (
        await session.execute(
            select(
                func.count(RequestLog.id),
                func.coalesce(func.sum(RequestLog.input_tokens), 0),
                func.coalesce(func.sum(RequestLog.output_tokens), 0),
                func.coalesce(func.sum(RequestLog.estimated_cost), 0),
            )
        )
    ).one()
    return {
        "requests": row[0],
        "input_tokens": row[1],
        "output_tokens": row[2],
        "estimated_cost": float(row[3]),
    }
