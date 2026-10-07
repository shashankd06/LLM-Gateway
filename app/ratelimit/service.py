from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import ApiKey, DailyUsage, RequestLog
from app.errors import GatewayError
from app.ratelimit.token_bucket import consume, until_midnight, utc


class DatabaseRateLimiter:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.sessions = sessions
        self.clock = clock

    async def now(self, session: AsyncSession) -> datetime:
        if self.clock is not None:
            return utc(self.clock())
        if session.get_bind().dialect.name == "postgresql":
            # Read after locking: transaction-start time would be stale after a lock wait.
            value = await session.scalar(select(func.clock_timestamp()))
            assert isinstance(value, datetime)
            return utc(value)
        return datetime.now(UTC)

    async def lock_key(
        self, session: AsyncSession, key_id: UUID, require_active: bool = True
    ) -> ApiKey:
        if session.get_bind().dialect.name == "sqlite":
            # Test-only backend: SQLite has no row locks, so serialize writers before reading.
            await session.execute(text("BEGIN IMMEDIATE"))
        key = await session.scalar(select(ApiKey).where(ApiKey.id == key_id).with_for_update())
        if key is None or (require_active and not key.active):
            raise GatewayError(401, "A valid bearer API key is required", "invalid_api_key")
        return key

    async def admit(self, log: RequestLog, reserved_tokens: int) -> None:
        """Commit the rate debit, budget hold, and pending log in one short transaction."""
        assert log.api_key_id is not None
        if reserved_tokens <= 0:
            raise ValueError("reserved_tokens must be positive")
        async with self.sessions() as session, session.begin():
            key = await self.lock_key(session, log.api_key_id)
            now = await self.now(session)
            decision = consume(
                key.bucket_tokens, key.bucket_updated_at, key.requests_per_minute, now
            )
            if not decision.allowed:
                raise GatewayError(
                    429, "Request rate limit exceeded", "rate_limit_exceeded", decision.retry_after
                )
            usage = await session.get(DailyUsage, (key.id, now.date()))
            charged = usage.tokens_charged if usage is not None else 0
            if charged + reserved_tokens > key.daily_token_budget:
                raise GatewayError(
                    429, "Daily token budget exceeded", "daily_budget_exceeded", until_midnight(now)
                )
            if usage is None:
                usage = DailyUsage(api_key_id=key.id, day=now.date(), tokens_charged=0)
                session.add(usage)
            usage.tokens_charged += reserved_tokens
            key.bucket_tokens = decision.tokens
            key.bucket_updated_at = decision.updated_at
            log.budget_day, log.budget_tokens = now.date(), reserved_tokens
            log.status = "in_progress"
            session.add(log)

    async def settle(self, outcome: RequestLog, usage_known: bool) -> bool:
        """Replace a hold exactly once, atomically with the final request log.

        Failed/crashed calls retain the hold because providers may have done unreported work.
        Day-specific rows ensure that late completions cannot debit a new day's budget.
        """
        assert outcome.api_key_id is not None
        async with self.sessions() as session, session.begin():
            key = await self.lock_key(session, outcome.api_key_id, require_active=False)
            stored = await session.get(RequestLog, outcome.id)
            if stored is None:
                raise RuntimeError("An admitted request log is required for settlement")
            if stored.status != "in_progress":
                return stored.status != "budget_overrun"
            assert stored.budget_day is not None
            usage = await session.get(DailyUsage, (key.id, stored.budget_day))
            assert usage is not None
            charge = (
                outcome.input_tokens + outcome.output_tokens
                if usage_known
                else stored.budget_tokens
            )
            usage.tokens_charged += charge - stored.budget_tokens
            stored.budget_tokens = charge
            stored.input_tokens, stored.output_tokens = outcome.input_tokens, outcome.output_tokens
            stored.latency_ms, stored.estimated_cost = outcome.latency_ms, outcome.estimated_cost
            stored.status, stored.http_status = outcome.status, outcome.http_status
            within_budget = usage.tokens_charged <= key.daily_token_budget
            if not within_budget:
                # An opaque provider tokenizer can exceed the conservative input estimate.
                # Charge all reported usage honestly, reject the response, and block further calls.
                stored.status, stored.http_status = "budget_overrun", 429
            outcome.status, outcome.http_status = stored.status, stored.http_status
            return within_budget
