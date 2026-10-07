import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError

from app.providers.base import GenerationParams, ProviderError, ProviderResponse, ProviderTimeout
from app.routing.circuit_breaker import CircuitBreaker, CircuitOpen, Permit
from app.routing.config import RoutingConfig, load_routing_config
from app.routing.router import ModelRouter, RouteTrace, confidence_failure
from tests.conftest import FakeProvider


class ScriptedProvider(FakeProvider):
    def __init__(self, name: str = "ollama") -> None:
        super().__init__()
        self.name = name
        self.outcomes: dict[str, list[str | BaseException]] = {}

    async def generate(self, prompt: str, model: str, params: GenerationParams) -> ProviderResponse:
        result = await super().generate(prompt, model, params)
        values = self.outcomes.get(model, [])
        outcome = values.pop(0) if values else "A sufficiently detailed answer."
        if isinstance(outcome, BaseException):
            raise outcome
        return result.model_copy(update={"text": outcome})


def routing_config(attempts: int = 2, threshold: int = 5) -> RoutingConfig:
    return RoutingConfig.model_validate(
        {
            "tiers": {
                "cheap": {"provider": "ollama", "model": "small", "timeout_seconds": 1},
                "strong": {"provider": "openai_compat", "model": "large", "timeout_seconds": 1},
            },
            "retry": {
                "max_attempts": attempts,
                "backoff_base_seconds": 0.1,
                "backoff_max_seconds": 0.2,
            },
            "circuit_breaker": {"failure_threshold": threshold, "recovery_seconds": 10},
        }
    )


def model_router(
    cheap: ScriptedProvider,
    strong: ScriptedProvider,
    attempts: int = 2,
    threshold: int = 5,
) -> ModelRouter:
    return ModelRouter(
        routing_config(attempts, threshold),
        {"ollama": cheap, "openai_compat": strong},
        sleep=AsyncMock(),
        jitter=lambda low, high: high / 2,
    )


@pytest.mark.parametrize(
    ("text", "json_mode", "reason"),
    [
        ("", False, "empty_response"),
        ("   ", False, "empty_response"),
        ("Paris", False, "short_response"),
        ("I cannot help with that.", False, "refusal"),
        ("I'm sorry, I can't do that", False, "refusal"),
        ("As an AI I cannot", False, "refusal"),
        ("A sufficiently detailed answer.", False, None),
        ('{"x": 1}', True, None),
        ("{}", True, None),
        ("[]", True, "invalid_json"),
        ("null", True, "invalid_json"),
        ('{"x": NaN}', True, "invalid_json"),
        ("```json\n{}\n```", True, "invalid_json"),
        ("The phrase 'I cannot' is quoted here.", False, None),
    ],
)
def test_confidence_heuristic(text: str, json_mode: bool, reason: str | None) -> None:
    assert confidence_failure(text, GenerationParams(json_mode=json_mode)) == reason


@pytest.mark.parametrize(
    "value",
    [
        {"retry": {"max_attempts": 0}},
        {"retry": {"max_attempts": 6}},
        {"retry": {"max_attempts": True}},
        {"retry": {"backoff_base_seconds": 3}},
        {"tiers": {"cheap": {"model": "x", "provider": "typo"}}},
        {"tiers": {"cheap": {"model": "x", "timeout_seconds": 0}}},
        {"circuit_breaker": {"failure_threshold": 0}},
        {"confidence": {"min_response_chars": 0}},
        {"unrecognized": True},
    ],
)
def test_yaml_schema_rejects_invalid_configuration(value: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        RoutingConfig.model_validate(value)


def test_committed_yaml_configuration() -> None:
    config = load_routing_config("routing.yaml")
    assert config.tiers.cheap.model == "qwen2.5:1.5b"
    assert config.tiers.strong.model == "qwen2.5:7b"
    assert config.retry.max_attempts == 2


async def test_cheap_success_does_not_call_strong() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    router = model_router(cheap, strong)
    trace = RouteTrace()
    result = await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert result.text == "A sufficiently detailed answer."
    assert len(cheap.calls) == 1 and not strong.calls
    assert trace.target is not None and trace.target.tier == "cheap"
    assert trace.attempts == 1 and trace.budget_charge == 15 and trace.escalation_reason is None


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty_response"),
        ("Paris", "short_response"),
        ("I cannot assist.", "refusal"),
    ],
)
async def test_low_confidence_escalates_without_retry(text: str, reason: str) -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.outcomes["small"] = [text]
    router = model_router(cheap, strong)
    trace = RouteTrace()
    await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert len(cheap.calls) == len(strong.calls) == 1
    assert trace.escalation_reason == reason and trace.budget_charge == 30
    assert trace.target is not None and trace.target.tier == "strong"
    router.sleep.assert_not_awaited()


async def test_malformed_json_escalates_and_valid_short_json_does_not() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.outcomes["small"] = ["not JSON", "{}"]
    strong.outcomes["large"] = ['{"answer": "Paris"}']
    router = model_router(cheap, strong)
    trace = RouteTrace()
    result = await router.generate(
        router.plan(None), "prompt", GenerationParams(json_mode=True), trace, 100
    )
    assert trace.escalation_reason == "invalid_json" and result.text.startswith("{")
    await router.generate(
        router.plan(None), "prompt", GenerationParams(json_mode=True), RouteTrace(), 100
    )
    assert len(strong.calls) == 1


async def test_retries_with_exponential_backoff_full_jitter_and_cap() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.outcomes["small"] = [ProviderError("private") for _ in range(3)]
    router = model_router(cheap, strong, attempts=4)
    trace = RouteTrace()
    await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert len(cheap.calls) == 4 and not strong.calls
    assert [call.args[0] for call in router.sleep.await_args_list] == [0.05, 0.1, 0.1]
    assert trace.budget_charge == 315 and trace.input_tokens == 12 and trace.output_tokens == 3


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (ProviderError("private"), "provider_error"),
        (ProviderTimeout("private"), "provider_timeout"),
    ],
)
async def test_exhausted_retries_fall_back_and_keep_unknown_allowances(
    failure: ProviderError,
    reason: str,
) -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.outcomes["small"] = [failure, failure]
    router = model_router(cheap, strong)
    trace = RouteTrace()
    await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert trace.escalation_reason == reason and trace.attempts == 3
    assert trace.unknown_tokens == 200 and trace.budget_charge == 215


async def test_attempt_deadline_escalates_even_if_adapter_never_returns() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.gate = asyncio.Event()
    config = routing_config(attempts=1).model_dump()
    config["tiers"]["cheap"]["timeout_seconds"] = 0.01
    router = ModelRouter(
        RoutingConfig.model_validate(config), {"ollama": cheap, "openai_compat": strong}
    )
    trace = RouteTrace()
    await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert trace.escalation_reason == "provider_timeout" and trace.unknown_tokens == 100


async def test_open_cheap_provider_skips_inference_and_uses_strong() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    router = model_router(cheap, strong, threshold=1)
    breaker = router.breakers["ollama"]
    await breaker.failure(await breaker.acquire())
    trace = RouteTrace()
    await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert not cheap.calls and len(strong.calls) == 1
    assert trace.escalation_reason == "circuit_open" and trace.budget_charge == 15


async def test_same_provider_circuit_is_shared_between_tiers() -> None:
    cheap = ScriptedProvider()
    config = routing_config(threshold=1).model_dump()
    config["tiers"]["strong"]["provider"] = "ollama"
    router = ModelRouter(RoutingConfig.model_validate(config), {"ollama": cheap}, sleep=AsyncMock())
    cheap.outcomes["small"] = [ProviderError("unavailable")]
    trace = RouteTrace()
    with pytest.raises(CircuitOpen):
        await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert len(cheap.calls) == 1 and trace.budget_charge == 100


@pytest.mark.parametrize("answer", ["", "not JSON", "[]", '{"value": Infinity}'])
async def test_invalid_strong_json_is_rejected(answer: str) -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    strong.outcomes["large"] = [answer]
    router = model_router(cheap, strong)
    trace = RouteTrace()
    with pytest.raises(ProviderError):
        await router.generate(
            router.plan("strong"), "prompt", GenerationParams(json_mode=True), trace, 100
        )
    assert trace.budget_charge == 15 and trace.attempts == 1


async def test_unexpected_errors_and_cancellation_are_not_retried() -> None:
    for failure in (RuntimeError("bug"), asyncio.CancelledError()):
        cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
        cheap.outcomes["small"] = [failure]
        router = model_router(cheap, strong)
        trace = RouteTrace()
        with pytest.raises(type(failure)):
            await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
        assert len(cheap.calls) == 1 and not strong.calls and trace.budget_charge == 100
        assert router.breakers["ollama"].failures == 0


def test_route_selection_and_cache_policy() -> None:
    router = model_router(ScriptedProvider(), ScriptedProvider("openai_compat"))
    assert router.plan(None) == router.plan("auto") == router.plan("cheap")
    assert [target.tier for target in router.plan("strong").targets] == ["strong"]
    assert router.plan("custom-model").targets[0].tier == "direct"
    assert router.plan(None).policy != router.plan("strong").policy
    with pytest.raises(ValueError, match="not enabled"):
        ModelRouter(routing_config(), {"ollama": ScriptedProvider()})


async def test_circuit_cooldown_single_concurrent_probe_and_recovery() -> None:
    now = Mock(return_value=0.0)
    breaker = CircuitBreaker(2, 10, now)
    await breaker.failure(await breaker.acquire())
    assert breaker.state == "closed"
    await breaker.failure(await breaker.acquire())
    with pytest.raises(CircuitOpen) as exc:
        await breaker.acquire()
    assert exc.value.retry_after == 10
    now.return_value = 10.0
    results = await asyncio.gather(*[breaker.acquire() for _ in range(40)], return_exceptions=True)
    permits = [result for result in results if not isinstance(result, BaseException)]
    assert len(permits) == 1 and sum(isinstance(result, CircuitOpen) for result in results) == 39
    await breaker.success(permits[0])
    assert breaker.state == "closed" and breaker.failures == 0
    await breaker.acquire()


async def test_stale_completions_cannot_reset_open_breaker() -> None:
    breaker = CircuitBreaker(1, 10)
    stale = await breaker.acquire()
    await breaker.failure(await breaker.acquire())
    await breaker.success(stale)
    await breaker.failure(stale)
    assert breaker.state == "open" and breaker.failures == 1


async def test_failed_or_cancelled_recovery_probe_reopens() -> None:
    for cancel in (False, True):
        now = Mock(return_value=0.0)
        breaker = CircuitBreaker(1, 10, now)
        await breaker.failure(await breaker.acquire())
        now.return_value = 10.0
        probe = await breaker.acquire()
        await (breaker.cancel(probe) if cancel else breaker.failure(probe))
        assert breaker.state == "open"
        with pytest.raises(CircuitOpen):
            await breaker.acquire()


async def test_cancel_after_response_does_not_strand_half_open_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    router = model_router(cheap, strong, threshold=1)
    breaker = router.breakers["ollama"]
    now = Mock(return_value=0.0)
    breaker.clock = now
    await breaker.failure(await breaker.acquire())
    now.return_value = 10.0
    started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_success = breaker.success

    async def delayed_success(permit: Permit) -> None:
        started.set()
        await release.wait()
        await original_success(permit)
        completed.set()

    monkeypatch.setattr(breaker, "success", delayed_success)
    trace = RouteTrace()
    task = asyncio.create_task(
        router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    )
    try:
        async with asyncio.timeout(5):
            await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        release.set()
        async with asyncio.timeout(5):
            await completed.wait()
        assert breaker.state == "closed" and trace.budget_charge == 15
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_cancel_during_retry_backoff_does_not_start_fallback() -> None:
    cheap, strong = ScriptedProvider(), ScriptedProvider("openai_compat")
    cheap.outcomes["small"] = [ProviderError("private")]
    router = model_router(cheap, strong)
    router.sleep = AsyncMock(side_effect=asyncio.CancelledError())
    trace = RouteTrace()
    with pytest.raises(asyncio.CancelledError):
        await router.generate(router.plan(None), "prompt", GenerationParams(), trace, 100)
    assert len(cheap.calls) == 1 and not strong.calls and trace.budget_charge == 100
