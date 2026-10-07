"""Routing outcome metadata and cache provenance."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("request_logs", sa.Column("tier", sa.String(10), nullable=True))
    op.add_column("request_logs", sa.Column("escalation_reason", sa.String(40), nullable=True))
    op.add_column(
        "request_logs", sa.Column("attempts", sa.Integer(), server_default="0", nullable=False)
    )
    op.alter_column("request_logs", "attempts", server_default=None)
    op.create_check_constraint("ck_request_attempts_nonnegative", "request_logs", "attempts >= 0")
    # v2 cache namespaces invalidate old rows; defaults preserve migration of historical entries.
    op.add_column(
        "cache_entries",
        sa.Column("provider", sa.String(40), server_default="legacy", nullable=False),
    )
    op.add_column(
        "cache_entries", sa.Column("tier", sa.String(10), server_default="direct", nullable=False)
    )
    op.alter_column("cache_entries", "provider", server_default=None)
    op.alter_column("cache_entries", "tier", server_default=None)


def downgrade() -> None:
    op.drop_column("cache_entries", "tier")
    op.drop_column("cache_entries", "provider")
    op.drop_constraint("ck_request_attempts_nonnegative", "request_logs", type_="check")
    op.drop_column("request_logs", "attempts")
    op.drop_column("request_logs", "escalation_reason")
    op.drop_column("request_logs", "tier")
