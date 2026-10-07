from datetime import UTC, datetime, timedelta, timezone

import pytest

from app.ratelimit.token_bucket import consume, until_midnight

NOW = datetime(2026, 10, 7, 12, tzinfo=UTC)


def test_initial_burst_is_bounded() -> None:
    tokens = 5.0
    for _ in range(5):
        decision = consume(tokens, NOW, 5, NOW)
        assert decision.allowed
        tokens = decision.tokens
    exhausted = consume(tokens, NOW, 5, NOW)
    assert not exhausted.allowed and exhausted.tokens == 0
    assert exhausted.retry_after == 12


def test_refill_rate_and_fractional_retry() -> None:
    decision = consume(0, NOW, 5, NOW + timedelta(seconds=6))
    assert not decision.allowed and decision.tokens == 0.5 and decision.retry_after == 6
    recovered = consume(decision.tokens, decision.updated_at, 5, NOW + timedelta(seconds=12))
    assert recovered.allowed and recovered.tokens == 0


def test_refill_cannot_accumulate_above_capacity() -> None:
    decision = consume(0, NOW, 5, NOW + timedelta(days=1))
    assert decision.allowed and decision.tokens == 4


def test_clock_regression_does_not_mint_tokens() -> None:
    earlier = consume(0, NOW, 5, NOW - timedelta(seconds=30))
    assert not earlier.allowed and earlier.updated_at == NOW and earlier.tokens == 0
    later = consume(earlier.tokens, earlier.updated_at, 5, NOW + timedelta(seconds=6))
    assert not later.allowed and later.tokens == 0.5


def test_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        consume(0, NOW, 0, NOW)


def test_utc_midnight_retry() -> None:
    assert until_midnight(NOW) == 43200
    assert until_midnight(NOW.replace(tzinfo=None)) == 43200
    assert until_midnight(datetime(2026, 10, 7, 23, 59, 59, 900000, tzinfo=UTC)) == 1
    india_time = datetime(2026, 10, 7, 17, 30, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    assert until_midnight(india_time) == 43200
