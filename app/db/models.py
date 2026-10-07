from datetime import date, datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    Uuid,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ApiKey(Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        CheckConstraint("requests_per_minute > 0", name="ck_api_key_rpm_positive"),
        CheckConstraint("daily_token_budget > 0", name="ck_api_key_budget_positive"),
        CheckConstraint(
            "bucket_tokens >= 0 AND bucket_tokens <= requests_per_minute",
            name="ck_api_key_bucket_bounds",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(100))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True)
    active: Mapped[bool] = mapped_column(default=True)
    requests_per_minute: Mapped[int]
    daily_token_budget: Mapped[int] = mapped_column(BigInteger)
    bucket_tokens: Mapped[float]
    bucket_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class DailyUsage(Base):
    __tablename__ = "daily_usage"
    __table_args__ = (CheckConstraint("tokens_charged >= 0", name="ck_daily_usage_nonnegative"),)

    api_key_id: Mapped[UUID] = mapped_column(Uuid, ForeignKey("api_keys.id"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    # Includes holds for in-flight requests. Finalization replaces each hold with actual usage.
    tokens_charged: Mapped[int] = mapped_column(BigInteger, default=0)


class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = (
        CheckConstraint("input_tokens >= 0 AND output_tokens >= 0", name="ck_tokens_nonnegative"),
        CheckConstraint("latency_ms >= 0", name="ck_latency_nonnegative"),
        CheckConstraint("estimated_cost >= 0", name="ck_cost_nonnegative"),
        CheckConstraint("cache_status IN ('hit', 'miss', 'bypass')", name="ck_cache_status"),
        CheckConstraint("budget_tokens >= 0", name="ck_request_budget_nonnegative"),
        Index("ix_request_logs_created_at", "created_at"),
        Index("ix_request_logs_api_key_created_at", "api_key_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    # Historical anonymous requests and rejected unknown keys remain nullable.
    api_key_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("api_keys.id", name="fk_request_logs_api_key"), nullable=True
    )
    budget_day: Mapped[date | None] = mapped_column(Date, nullable=True)
    budget_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    model: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(40))
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    latency_ms: Mapped[float]
    cache_status: Mapped[str] = mapped_column(String(10))
    estimated_cost: Mapped[Decimal] = mapped_column(Numeric(20, 10), default=Decimal("0"))
    status: Mapped[str] = mapped_column(String(40))
    http_status: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
