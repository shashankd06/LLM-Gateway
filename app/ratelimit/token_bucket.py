from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from math import ceil


def utc(value: datetime) -> datetime:
    """SQLite drops timezone information; PostgreSQL returns aware timestamps."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class BucketDecision:
    allowed: bool
    tokens: float
    updated_at: datetime
    retry_after: int


def consume(
    tokens: float, updated_at: datetime, requests_per_minute: int, now: datetime
) -> BucketDecision:
    """Pure calculation: callers must serialize read/modify/write of persisted state."""
    if requests_per_minute <= 0:
        raise ValueError("requests_per_minute must be positive")
    previous = utc(updated_at)
    current = max(utc(now), previous)
    available = min(
        float(requests_per_minute),
        tokens + (current - previous).total_seconds() * requests_per_minute / 60,
    )
    if available >= 1:
        return BucketDecision(True, available - 1, current, 0)
    retry_after = max(1, ceil((1 - available) * 60 / requests_per_minute))
    return BucketDecision(False, available, current, retry_after)


def until_midnight(now: datetime) -> int:
    current = utc(now)
    midnight = datetime.combine(current.date() + timedelta(days=1), time(), tzinfo=UTC)
    return max(1, ceil((midnight - current).total_seconds()))
