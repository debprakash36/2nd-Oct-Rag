"""query log

Revision ID: 0002_query_log
Revises: 0001_init
Create Date: 2026-10-01

Adds `query_logs` (FR-28). Every field listed in architecture.md 5 is created in
one step on purpose: retroactive instrumentation is impossible, and the fields
that distinguish a content gap from a retrieval gap have to exist before the
first production query, not after the first incident.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_query_log"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "query_logs",
        sa.Column("query_id", sa.String(length=32), primary_key=True),
        sa.Column("trace_id", sa.String(length=32), nullable=True),
        sa.Column("conversation_id", sa.String(length=32), nullable=True),
        sa.Column("original_query", sa.Text(), nullable=False),
        sa.Column("rewritten_query", sa.Text(), nullable=False),
        sa.Column("retrieved_ids", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("threshold_applied", sa.Float(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=32), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ttft_ms", sa.Integer(), nullable=True),
        sa.Column("total_ms", sa.Integer(), nullable=True),
        sa.Column("citations_stripped", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("feedback", sa.String(length=8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_query_logs_trace", "query_logs", ["trace_id"])
    op.create_index("ix_query_logs_conversation", "query_logs", ["conversation_id"])
    op.create_index("ix_query_logs_created", "query_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_query_logs_created", table_name="query_logs")
    op.drop_index("ix_query_logs_conversation", table_name="query_logs")
    op.drop_index("ix_query_logs_trace", table_name="query_logs")
    op.drop_table("query_logs")
