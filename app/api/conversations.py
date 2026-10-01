"""Conversation lifecycle (FR-25, FR-23, architecture.md 6.1).

New chat, list, read a thread, delete a thread. The chat endpoint in
`app/api/chat.py` owns writing turns; this module owns the conversation itself.

No `owner` field and no auth check anywhere in here, because v1 has no
authentication (architecture.md NG5). That is a deliberate, documented gap: when
auth lands, this router and every `get_conversation` call site needs an ownership
predicate, and the absence of one here means "first public deployment exposes every
conversation", not "ownership is handled". Do not build a public deployment on this
without reading that section.

Titles are *derived*, not stored. The first user message is a good enough label and
storing it would add a column that goes stale the moment the user edits or deletes
that first turn. Deriving it per request costs one query that the list view needs
anyway for `turn_count`.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, Query, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import conversation as conv_store
from app.db.models import Turn, TurnRole
from app.db.session import get_db

router = APIRouter(prefix="/conversations", tags=["conversations"])

#: Long enough to be recognisable in a sidebar, short enough not to wrap every
#: entry onto three lines. Truncation is on a word boundary when one exists nearby,
#: so a list does not show half a word at the end.
PREVIEW_CHARS = 72

#: `recent_turns` takes a limit, but reading a thread is not a context-window read
#: and must return every turn. Rather than add a second read path with the same
#: descending-then-reverse logic, pass a sentinel no real limit can beat.
_NO_LIMIT = 1_000_000


class TurnOut(BaseModel):
    """One persisted turn."""

    turn_id: str
    turn_index: int
    role: str
    content: str
    citations: list[str] = Field(default_factory=list)
    query_id: str | None = None
    abstained: bool = False
    created_at: dt.datetime


class ConversationOut(BaseModel):
    """A conversation as shown in the list view."""

    conversation_id: str
    created_at: dt.datetime
    updated_at: dt.datetime
    turn_count: int
    preview: str


class ConversationDetail(ConversationOut):
    """A conversation including its full turn history."""

    turns: list[TurnOut] = Field(default_factory=list)


def _preview(first_question: str | None) -> str:
    """A one-line label for the sidebar (FR-25)."""
    if not first_question:
        return "New conversation"
    collapsed = " ".join(first_question.split())
    if len(collapsed) <= PREVIEW_CHARS:
        return collapsed
    cut = collapsed[:PREVIEW_CHARS]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut + "…"


def _first_question(session: Session, conversation_id: str) -> str | None:
    return session.execute(
        select(Turn.content)
        .where(Turn.conversation_id == conversation_id, Turn.role == TurnRole.USER)
        .order_by(Turn.turn_index)
        .limit(1)
    ).scalar()


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
def new_conversation(session: Session = Depends(get_db)) -> ConversationOut:
    """Start an empty conversation (FR-25).

    Created eagerly by the UI rather than lazily on the first message, so the id
    that the client holds is the id the turns will be written under. The lazy
    alternative loses a turn to a lost response: the client retries a request whose
    server side already answered, and now there are two threads.
    """
    conversation = conv_store.create_conversation(session)
    session.commit()
    return ConversationOut(
        conversation_id=conversation.conversation_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        turn_count=0,
        preview=_preview(None),
    )


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> list[ConversationOut]:
    """Recent conversations, most recently active first."""
    conversations = conv_store.list_conversations(session, limit=limit, offset=offset)
    return [
        ConversationOut(
            conversation_id=c.conversation_id,
            created_at=c.created_at,
            updated_at=c.updated_at,
            turn_count=conv_store.turn_count(session, c.conversation_id),
            preview=_preview(_first_question(session, c.conversation_id)),
        )
        for c in conversations
    ]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def read_conversation(
    conversation_id: str, session: Session = Depends(get_db)
) -> ConversationDetail:
    """A conversation with its turns, oldest first.

    Raises `NotFoundError` for an unknown id, which the app-wide handler renders as
    404. Returning an empty thread instead would make a stale bookmark look like a
    new conversation, and the user would have no way to tell the two apart.
    """
    conversation = conv_store.get_conversation(session, conversation_id)
    turns = conv_store.recent_turns(session, conversation_id, limit=_NO_LIMIT)
    return ConversationDetail(
        conversation_id=conversation.conversation_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        turn_count=len(turns),
        preview=_preview(next((t.content for t in turns if t.role is TurnRole.USER), None)),
        turns=[
            TurnOut(
                turn_id=t.turn_id,
                turn_index=t.turn_index,
                role=str(t.role),
                content=t.content,
                citations=list(t.citations or []),
                query_id=t.query_id,
                abstained=t.abstained,
                created_at=t.created_at,
            )
            for t in turns
        ],
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str, session: Session = Depends(get_db)
) -> Response:
    """Delete a conversation and its turns (FR-25).

    204 with no body. `query_logs` rows are intentionally retained; see
    `conversation.delete_conversation` for why.
    """
    conv_store.delete_conversation(session, conversation_id)
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)