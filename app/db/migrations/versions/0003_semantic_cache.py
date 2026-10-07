"""Key-scoped exact and semantic cache with vector(384) embeddings."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "cache_entries",
        sa.Column("id", sa.Uuid(), primary_key=True, nullable=False),
        sa.Column("api_key_id", sa.Uuid(), sa.ForeignKey("api_keys.id"), nullable=False),
        sa.Column("namespace", sa.String(64), nullable=False),
        sa.Column("model", sa.String(200), nullable=False),
        sa.Column("params_hash", sa.String(64), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("context_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=False),
        sa.Column("finish_reason", sa.String(10), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "api_key_id", "namespace", "model", "params_hash", "prompt_hash", name="uq_cache_exact"
        ),
    )
    op.create_index(
        "ix_cache_scope",
        "cache_entries",
        ["namespace", "model", "params_hash", "context_hash", "api_key_id"],
    )
    op.create_index("ix_cache_expiry", "cache_entries", ["expires_at"])
    # Exact nearest-neighbor search is deterministic at portfolio scale. Add HNSW only after
    # measuring corpus growth; approximate search can miss valid hits after tenant filtering.


def downgrade() -> None:
    op.drop_table("cache_entries")
