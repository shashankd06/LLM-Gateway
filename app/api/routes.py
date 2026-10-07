import asyncio
import hashlib
import json
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
from app.db.models import RequestLog
from app.deps import ApiKeyDep, ProviderDep, RateLimiterDep, SessionDep, SettingsDep
from app.errors import GatewayError
from app.metrics.tracking import estimate_cost
from app.providers.base import ProviderError, ProviderTimeout
from app.ratelimit.token_bucket import until_midnight

router = APIRouter()


def prompt_hash(messages: list[dict[str, str]]) -> str:
    # Hash structured messages so role/content boundaries cannot collide.
    canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def error_response(status_code: int, message: str, code: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": code, "param": None, "code": code}},
    )


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    response: Response,
    settings: SettingsDep,
    provider: ProviderDep,
    key: ApiKeyDep,
    limiter: RateLimiterDep,
) -> ChatCompletionResponse | JSONResponse:
    model = body.model or settings.default_model
    messages = [message.model_dump() for message in body.messages]
    # The prompt-based adapter contract preserves text history but not native chat-role semantics.
    prompt = "\n\n".join(f"[{message.role}]\n{message.content}" for message in body.messages)
    prompt += "\n\n[assistant]\n"
    params = body.generation_params()
    params.max_tokens = params.max_tokens or settings.default_max_output_tokens
    if params.max_tokens > settings.max_output_tokens:
        raise GatewayError(400, "max_tokens exceeds the gateway limit", "invalid_request_error")
    # Byte count + template padding errs high for common tokenizers; actual usage is reconciled.
    reserved_tokens = (
        len(prompt.encode("utf-8")) + settings.budget_input_token_padding + params.max_tokens
    )
    log = RequestLog(
        id=uuid4(),
        api_key_id=key.id,
        model=model,
        provider=provider.name,
        prompt_hash=prompt_hash(messages),
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        cache_status="bypass",
        estimated_cost=0,
        status="success",
        http_status=200,
    )
    await limiter.admit(log, reserved_tokens)
    log.status = "success"
    failure: JSONResponse | None = None
    result = None
    try:
        result = await provider.generate(prompt, model, params)
        log.input_tokens = result.input_tokens
        log.output_tokens = result.output_tokens
        log.estimated_cost = estimate_cost(result.input_tokens, result.output_tokens, settings)
    except asyncio.CancelledError:
        log.status, log.http_status = "cancelled", 499
        log.latency_ms = (perf_counter() - request.state.started) * 1000
        await asyncio.shield(limiter.settle(log, usage_known=False))
        raise
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
    log.latency_ms = (perf_counter() - request.state.started) * 1000
    within_budget = await limiter.settle(log, usage_known=result is not None)
    if not within_budget:
        failure = error_response(
            429, "Provider usage exceeded the budget reservation", "daily_budget_exceeded"
        )
        # The relevant hold belongs to its original UTC day. Retry uses the next UTC reset.
        async with limiter.sessions() as session:
            failure.headers["Retry-After"] = str(until_midnight(await limiter.now(session)))
    if failure is not None:
        failure.headers["X-Request-ID"] = str(log.id)
        failure.headers["X-Cache"] = "BYPASS"
        return failure
    assert result is not None
    response.headers["X-Request-ID"] = str(log.id)
    response.headers["X-Cache"] = "BYPASS"
    return ChatCompletionResponse(
        id=f"chatcmpl-{log.id}",
        created=int(time()),
        model=model,
        choices=[
            Choice(
                message=AssistantMessage(content=result.text), finish_reason=result.finish_reason
            )
        ],
        usage=Usage(
            prompt_tokens=result.input_tokens,
            completion_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
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
