"""conversations and turns

Revision ID: 0003_conversations
Revises: 0002_query_log
Create Date: 2026-10-01

Adds the conversation store (FR-23, architecture.md 5). `query_logs.conversation_id`
already existed as a bare string from Phase 3 and is left alone deliberately: it is
a correlation key, and adding a foreign key here would couple a turn's persistence
to the streaming generator's write order (see the column comment in models.py).

`turn_index` is a monotonic counter rather than ordering on `turn_id`. The id is a
random uuid, so ordering by it would return a thread in a different order per read
and the last-N-turns context window would contain an arbitrary slice of it.
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003_conversations"
down_revision = "0002_query_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "conversations",
        sa.Column("conversation_id", sa.String(length=32), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_conversations_created", "conversations", ["created_at"])

    op.create_table(
        "turns",
        sa.Column("turn_id", sa.String(length=32), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(length=32),
            sa.ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("turn_index", sa.Integer(), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", name="turnrole", native_enum=False,
                                  length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("query_id", sa.String(length=32), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # Mirrors the model constraint. Named explicitly so the DB rejects a
        # duplicate index rather than the application discovering it at read time,
        # where it would look like an intermittent ordering bug.
        sa.UniqueConstraint("conversation_id", "turn_index", name="uq_turn_conversation_index"),
    )
    op.create_index("ix_turns_conversation", "turns", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_turns_conversation", table_name="turns")
    op.drop_table("turns")
    op.drop_index("ix_conversations_created", table_name="conversations")
    op.drop_table("conversations")
