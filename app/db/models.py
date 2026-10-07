from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, Numeric, String, Uuid, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    __tablename__ = "request_logs"
    __table_args__ = (
        CheckConstraint("input_tokens >= 0 AND output_tokens >= 0", name="ck_tokens_nonnegative"),
        CheckConstraint("latency_ms >= 0", name="ck_latency_nonnegative"),
        CheckConstraint("estimated_cost >= 0", name="ck_cost_nonnegative"),
        CheckConstraint("cache_status IN ('hit', 'miss', 'bypass')", name="ck_cache_status"),
        Index("ix_request_logs_created_at", "created_at"),
        Index("ix_request_logs_api_key_created_at", "api_key_id", "created_at"),
    )

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    # Phase 2 will add the API-key table and foreign key; anonymous Phase 1 calls are null.
    api_key_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
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
