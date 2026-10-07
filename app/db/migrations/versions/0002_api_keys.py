"""API keys, shared token buckets, and daily token-budget holds."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("requests_per_minute", sa.Integer(), nullable=False),
        sa.Column("daily_token_budget", sa.BigInteger(), nullable=False),
        sa.Column("bucket_tokens", sa.Float(), nullable=False),
        sa.Column("bucket_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("requests_per_minute > 0", name="ck_api_key_rpm_positive"),
        sa.CheckConstraint("daily_token_budget > 0", name="ck_api_key_budget_positive"),
        sa.CheckConstraint(
            "bucket_tokens >= 0 AND bucket_tokens <= requests_per_minute",
            name="ck_api_key_bucket_bounds",
        ),
    )
    op.create_table(
        "daily_usage",
        sa.Column(
            "api_key_id", sa.Uuid(), sa.ForeignKey("api_keys.id"), primary_key=True, nullable=False
        ),
        sa.Column("day", sa.Date(), primary_key=True, nullable=False),
        sa.Column("tokens_charged", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("tokens_charged >= 0", name="ck_daily_usage_nonnegative"),
    )
    op.add_column("request_logs", sa.Column("budget_day", sa.Date(), nullable=True))
    # The temporary default preserves existing Phase 1 rows, then application code owns defaults.
    op.add_column(
        "request_logs",
        sa.Column("budget_tokens", sa.BigInteger(), nullable=False, server_default="0"),
    )
    op.alter_column("request_logs", "budget_tokens", server_default=None)
    op.create_check_constraint(
        "ck_request_budget_nonnegative", "request_logs", "budget_tokens >= 0"
    )
    op.create_foreign_key(
        "fk_request_logs_api_key", "request_logs", "api_keys", ["api_key_id"], ["id"]
    )


def downgrade() -> None:
    op.drop_constraint("fk_request_logs_api_key", "request_logs", type_="foreignkey")
    op.drop_constraint("ck_request_budget_nonnegative", "request_logs", type_="check")
    op.drop_column("request_logs", "budget_tokens")
    op.drop_column("request_logs", "budget_day")
    op.drop_table("daily_usage")
    op.drop_table("api_keys")
