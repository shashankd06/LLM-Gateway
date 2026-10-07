"""Enable pgvector and create request logs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("api_key_id", sa.Uuid(), nullable=True),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("provider", sa.String(40), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("cache_status", sa.String(10), nullable=False),
        sa.Column("estimated_cost", sa.Numeric(20, 10), nullable=False),
        sa.Column("status", sa.String(40), nullable=False),
        sa.Column("http_status", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "input_tokens >= 0 AND output_tokens >= 0", name="ck_tokens_nonnegative"
        ),
        sa.CheckConstraint("latency_ms >= 0", name="ck_latency_nonnegative"),
        sa.CheckConstraint("estimated_cost >= 0", name="ck_cost_nonnegative"),
        sa.CheckConstraint("cache_status IN ('hit', 'miss', 'bypass')", name="ck_cache_status"),
    )
    op.create_index("ix_request_logs_created_at", "request_logs", ["created_at"])
    op.create_index(
        "ix_request_logs_api_key_created_at", "request_logs", ["api_key_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_table("request_logs")
    # Leave vector installed: another application may also own data that depends on it.
