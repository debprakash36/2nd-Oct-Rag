"""Conversation and turn persistence (FR-23, FR-25, architecture.md 5).

A conversation is a thread of turns. Turns exist for two consumers that want
*different* subsets, and conflating them is the easy mistake here:

* The **model** needs both roles. "How long is *that*?" is only resolvable if the
  model can see what it said last turn, so `message_history` returns interleaved
  user/assistant turns in order.
* The **retriever** needs user turns only. `rewrite.extract_topic` reduces a turn to
  a noun phrase by stripping interrogative leads, so feeding it a full assistant
  paragraph yields a mangled antecedent and a confidently wrong standalone query.
  `query_history` returns bare user questions.

Neither includes FR-24's running summary. Condensing older turns is deferred
(implementation.md §9.1), so the history window is a hard cut at the last N turns:
past that, anaphora resolution simply stops working, which is a visible and
correctable limitation rather than a silent one.

Turns are appended *after* the answer completes, from inside the streaming
generator, for the same reason `QueryLog` is written there: a client that
disconnects mid-answer must still leave the conversation and the log consistent
enough to be diagnosable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models import Conversation, Turn, TurnRole

log = get_logger("app.db.conversation")


def create_conversation(session: Session) -> Conversation:
    """Start an empty conversation (FR-25, "new chat").

    Flushed, not committed: the caller owns the transaction, matching
    `query_log.record_query`. The id is available immediately because it is
    generated on construction.
    """
    conversation = Conversation()
    session.add(conversation)
    session.flush()
    log.info("conversation created", extra={"conversation_id": conversation.conversation_id})
    return conversation


def get_conversation(session: Session, conversation_id: str) -> Conversation:
    """Fetch a conversation or raise `NotFoundError`.

    Raises rather than returning `None` so a caller cannot accidentally treat a
    missing conversation as an empty one and silently discard a user's thread.
    """
    conversation = session.get(Conversation, conversation_id)
    if conversation is None:
        raise NotFoundError("conversation")
    return conversation


def list_conversations(
    session: Session, *, limit: int = 50, offset: int = 0
) -> list[Conversation]:
    """Most recently updated first.

    Ordered by `updated_at` rather than `created_at` so a conversation the user is
    actively talking in stays at the top of the list; a thread they just came back
    to is the one they are looking for.
    """
    stmt = (
        select(Conversation)
        .order_by(Conversation.updated_at.desc(), Conversation.conversation_id)
        .limit(max(limit, 0))
        .offset(max(offset, 0))
    )
    return list(session.execute(stmt).scalars().all())


def delete_conversation(session: Session, conversation_id: str) -> None:
    """Delete a conversation and its turns (FR-25).

    Turns cascade at the database level (`ondelete="CASCADE"`) and again via the
    ORM relationship. SQLite only honours the FK when foreign keys are enabled on
    the connection, which they are not by default, so the explicit delete of turns
    below is what makes this reliable on SQLite rather than Postgres-only — the
    same reason `purge_chunks` deletes keyword rows explicitly.

    `query_logs` rows are deliberately **kept**. They are the FR-28 instrumentation
    substrate and the PRD's improvement loop reads them after the fact; deleting a
    user's thread is not a licence to erase the evidence that a bad answer was
    served. The orphaned `conversation_id` becomes a correlation key that matches
    nothing, which is harmless and reversible.
    """
    get_conversation(session, conversation_id)
    session.execute(delete(Turn).where(Turn.conversation_id == conversation_id))
    session.execute(
        delete(Conversation).where(Conversation.conversation_id == conversation_id)
    )
    session.flush()
    log.info("conversation deleted", extra={"conversation_id": conversation_id})


def next_turn_index(session: Session, conversation_id: str) -> int:
    """The index the next appended turn should take."""
    highest = session.execute(
        select(func.max(Turn.turn_index)).where(Turn.conversation_id == conversation_id)
    ).scalar()
    return 0 if highest is None else int(highest) + 1


def append_turn(
    session: Session,
    conversation_id: str,
    *,
    role: TurnRole | str,
    content: str,
    citations: Sequence[str] = (),
    query_id: str | None = None,
    abstained: bool = False,
) -> Turn:
    """Append one turn to a conversation.

    `role` accepts the enum or its string value so callers reading a persisted turn
    back can round-trip it without converting.

    The turn is flushed so the caller can commit a user turn and its assistant
    reply together. The `(conversation_id, turn_index)` unique constraint is the
    backstop: two concurrent appends compute the same index, and one of them is
    rejected by the database rather than producing a thread that silently loses a
    message when ordered by a duplicated index.
    """
    turn = Turn(
        conversation_id=conversation_id,
        turn_index=next_turn_index(session, conversation_id),
        role=TurnRole(role),
        content=content,
        citations=list(citations),
        query_id=query_id,
        abstained=abstained,
    )
    session.add(turn)
    # Touch the conversation so `list_conversations` orders by recency of activity
    # rather than recency of creation. Set explicitly rather than leaning on the
    # column's `onupdate`, which would work but only as a side effect of the ORM
    # noticing a dirty attribute — invisible at the call site.
    conversation = session.get(Conversation, conversation_id)
    if conversation is not None:
        conversation.updated_at = dt.datetime.now(dt.UTC)
    session.flush()
    return turn


def recent_turns(session: Session, conversation_id: str, *, limit: int) -> list[Turn]:
    """The last `limit` turns in order, oldest first.

    Read with a descending subquery and a re-sort because SQL has no "last N in
    ascending order" primitive: selecting the newest N and reversing is what makes
    the caller's history chronological, and an LLM given reversed history produces
    visibly confused anaphora resolution.
    """
    if limit <= 0:
        return []
    newest_first = (
        select(Turn)
        .where(Turn.conversation_id == conversation_id)
        .order_by(Turn.turn_index.desc())
        .limit(limit)
    )
    return list(reversed(list(session.execute(newest_first).scalars().all())))


def message_history(
    session: Session, conversation_id: str, *, limit: int
) -> list[dict[str, str]]:
    """Interleaved user/assistant turns as chat messages, for `build_messages`."""
    return [
        {"role": str(turn.role), "content": turn.content}
        for turn in recent_turns(session, conversation_id, limit=limit)
    ]


def query_history(session: Session, conversation_id: str, *, limit: int) -> list[str]:
    """Recent *user* questions, for rule-based anaphora resolution.

    Assistant turns excluded — see the module docstring. The caller passes
    `retrieval_memory_turns` (default 10). `resolve_anaphora` walks that window
    from the newest turn backward, so a short reply does not hide the topic.
    """
    if limit <= 0:
        return []
    stmt = (
        select(Turn.content)
        .where(Turn.conversation_id == conversation_id, Turn.role == TurnRole.USER)
        .order_by(Turn.turn_index.desc())
        .limit(limit)
    )
    contents = [row[0] for row in session.execute(stmt).all()]
    return list(reversed(contents))


def turn_count(session: Session, conversation_id: str) -> int:
    """Number of turns stored, for the conversation list view."""
    return int(
        session.execute(
            select(func.count(Turn.turn_id)).where(Turn.conversation_id == conversation_id)
        ).scalar_one()
    )
