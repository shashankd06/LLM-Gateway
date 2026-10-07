import asyncio
import json
import random
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from app.cache.semantic_cache import fingerprint
from app.providers.base import (
    GenerationParams,
    ProviderAdapter,
    ProviderError,
    ProviderResponse,
    ProviderTimeout,
)
from app.routing.circuit_breaker import CircuitBreaker, CircuitOpen
from app.routing.config import RoutingConfig

TierName = Literal["cheap", "strong", "direct"]
REFUSAL = re.compile(r"^(?:i(?:'m| am) sorry\b|i (?:cannot|can't|am unable to)\b|as an ai\b)", re.I)


def reject_json_constant(value: str) -> None:
    raise ValueError("Non-standard JSON constant")


def confidence_failure(text: str, params: GenerationParams, min_chars: int = 12) -> str | None:
    answer = text.strip()
    if not answer:
        return "empty_response"
    if params.json_mode:
        try:
            parsed = json.loads(answer, parse_constant=reject_json_constant)
        except ValueError:
            return "invalid_json"
        return None if isinstance(parsed, dict) else "invalid_json"
    if REFUSAL.match(answer):
        return "refusal"
    if len(answer) < min_chars:
        return "short_response"
    return None


@dataclass(frozen=True)
class Target:
    tier: TierName
    provider: str
    model: str
    timeout_seconds: float


@dataclass(frozen=True)
class RoutePlan:
    targets: tuple[Target, ...]
    max_attempts: int
    policy: str


@dataclass
class RouteTrace:
    target: Target | None = None
    attempts: int = 0
    escalation_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    unknown_tokens: int = 0

    @property
    def budget_charge(self) -> int:
        return self.input_tokens + self.output_tokens + self.unknown_tokens


class ModelRouter:
    def __init__(
        self,
        config: RoutingConfig,
        providers: dict[str, ProviderAdapter],
        default_provider: str = "ollama",
        default_model: str = "qwen2.5:1.5b",
        enabled: bool = True,
        default_timeout: float = 120,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.config, self.providers = config, providers
        self.default_provider, self.default_model = default_provider, default_model
        self.enabled, self.default_timeout = enabled, default_timeout
        self.sleep, self.jitter = sleep, jitter
        required = {default_provider}
        if enabled:
            required.update((config.tiers.cheap.provider, config.tiers.strong.provider))
        if not required <= providers.keys():
            raise ValueError("Routing references a provider that is not enabled")
        self.breakers = {
            name: CircuitBreaker(
                config.circuit_breaker.failure_threshold, config.circuit_breaker.recovery_seconds
            )
            for name in providers
        }

    def plan(self, requested_model: str | None) -> RoutePlan:
        if self.enabled and requested_model in (None, "auto", "cheap", "strong"):
            names: tuple[Literal["cheap", "strong"], ...] = (
                ("strong",) if requested_model == "strong" else ("cheap", "strong")
            )
            configured_targets: list[Target] = []
            for name in names:
                tier = getattr(self.config.tiers, name)
                configured_targets.append(
                    Target(name, tier.provider, tier.model, tier.timeout_seconds)
                )
            targets = tuple(configured_targets)
        else:
            targets = (
                Target(
                    "direct",
                    self.default_provider,
                    requested_model or self.default_model,
                    self.default_timeout,
                ),
            )
        attempts = self.config.retry.max_attempts if self.enabled else 1
        # Routing changes must not reuse entries made under a different quality policy.
        policy = fingerprint(
            [
                self.config.model_dump(mode="json"),
                self.enabled,
                [(t.tier, t.provider, t.model) for t in targets],
            ]
        )
        return RoutePlan(targets, attempts, policy)

    async def _attempt(
        self,
        target: Target,
        prompt: str,
        params: GenerationParams,
        trace: RouteTrace,
        attempt_allowance: int,
    ) -> ProviderResponse:
        breaker = self.breakers[target.provider]
        permit = await breaker.acquire() if self.enabled else None
        trace.target = target
        trace.attempts += 1
        # Record a hold before awaiting the provider: cancellation/crashes leave usage unknown.
        trace.unknown_tokens += attempt_allowance
        try:
            async with asyncio.timeout(target.timeout_seconds):
                result = await self.providers[target.provider].generate(
                    prompt, target.model, params
                )
        except TimeoutError as exc:
            if permit is not None:
                await asyncio.shield(breaker.failure(permit))
            raise ProviderTimeout("Provider attempt deadline elapsed") from exc
        except ProviderError:
            if permit is not None:
                await asyncio.shield(breaker.failure(permit))
            raise
        except BaseException:
            if permit is not None:
                await asyncio.shield(breaker.cancel(permit))
            raise
        trace.unknown_tokens -= attempt_allowance
        trace.input_tokens += result.input_tokens
        trace.output_tokens += result.output_tokens
        if permit is not None:
            # Disconnects must not interrupt release of an exclusive half-open probe.
            await asyncio.shield(breaker.success(permit))
        return result

    async def _generate_target(
        self,
        target: Target,
        plan: RoutePlan,
        prompt: str,
        params: GenerationParams,
        trace: RouteTrace,
        attempt_allowance: int,
    ) -> ProviderResponse:
        for attempt in range(plan.max_attempts):
            try:
                return await self._attempt(target, prompt, params, trace, attempt_allowance)
            except CircuitOpen:
                raise  # An open provider should never be retried during its cooldown.
            except ProviderError:
                if attempt + 1 == plan.max_attempts:
                    raise
                cap = min(
                    self.config.retry.backoff_max_seconds,
                    self.config.retry.backoff_base_seconds * 2**attempt,
                )
                await self.sleep(self.jitter(0, cap))
        raise AssertionError("Route plans require at least one attempt")

    async def generate(
        self,
        plan: RoutePlan,
        prompt: str,
        params: GenerationParams,
        trace: RouteTrace,
        attempt_allowance: int,
    ) -> ProviderResponse:
        for index, target in enumerate(plan.targets):
            try:
                result = await self._generate_target(
                    target, plan, prompt, params, trace, attempt_allowance
                )
            except ProviderError as exc:
                if index + 1 == len(plan.targets):
                    raise
                trace.escalation_reason = (
                    "circuit_open"
                    if isinstance(exc, CircuitOpen)
                    else "provider_timeout"
                    if isinstance(exc, ProviderTimeout)
                    else "provider_error"
                )
                continue
            if target.tier == "cheap":
                reason = confidence_failure(
                    result.text, params, self.config.confidence.min_response_chars
                )
                if reason is not None:
                    trace.escalation_reason = reason
                    continue  # A completed low-confidence answer is not retried on the same tier.
            elif target.tier == "strong" and (not result.text.strip() or params.json_mode):
                reason = confidence_failure(
                    result.text, params, self.config.confidence.min_response_chars
                )
                if reason in ("empty_response", "invalid_json"):
                    raise ProviderError("Strong tier returned invalid output")
            return result
        raise ProviderError("No tier produced an acceptable response")
