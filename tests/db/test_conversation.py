"""Conversation store tests (FR-23, FR-25).

The store is the thing that has to stay correct on its own, because the API layer
delegates to it and the chat endpoint's persistence is only as good as these
invariants.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db import conversation as store
from app.db.models import QueryLog, Turn, TurnRole


@pytest.fixture
def conv(session: Session) -> str:
    """An empty conversation, committed."""
    conversation = store.create_conversation(session)
    session.commit()
    return conversation.conversation_id


class TestLifecycle:
    def test_create_generates_id_and_timestamps(self, session: Session, conv: str):
        conversation = store.get_conversation(session, conv)
        assert conversation.conversation_id == conv
        assert conversation.created_at is not None
        assert conversation.updated_at is not None
        assert store.turn_count(session, conv) == 0

    def test_ids_are_unique(self, session: Session):
        ids = {store.create_conversation(session).conversation_id for _ in range(25)}
        assert len(ids) == 25

    def test_get_missing_raises_not_found(self, session: Session):
        # Raising rather than returning None: a None return invites a caller to
        # treat a missing thread as an empty one and silently start over.
        with pytest.raises(NotFoundError):
            store.get_conversation(session, "no-such-conversation")

    def test_delete_removes_turns(self, session: Session, conv: str):
        store.append_turn(session, conv, role=TurnRole.USER, content="hi")
        store.append_turn(session, conv, role=TurnRole.ASSISTANT, content="hello")
        session.commit()
        assert store.turn_count(session, conv) == 2

        store.delete_conversation(session, conv)
        session.commit()
        assert session.execute(
            select(Turn).where(Turn.conversation_id == conv)
        ).scalars().all() == []

    def test_delete_missing_raises_not_found(self, session: Session):
        with pytest.raises(NotFoundError):
            store.delete_conversation(session, "no-such-conversation")

    def test_delete_retains_query_logs(self, session: Session, conv: str):
        """QueryLog is the FR-28/FR-30 improvement-loop substrate.

        Deleting a user's thread must not erase the record that an answer was served
        there — the PRD's analysis reads those rows after the fact, and a user
        clearing their history would otherwise silently remove it from the dataset.
        """
        session.add(
            QueryLog(
                conversation_id=conv,
                original_query="what is the refund window?",
                rewritten_query="refund window days",
                retrieved_ids=[],
                scores={},
                threshold_applied=0.3,
                abstained=False,
                model="fake",
                prompt_version="chat.v1",
                tokens_in=10,
                tokens_out=20,
                citations_stripped=0,
            )
        )
        store.append_turn(session, conv, role=TurnRole.USER, content="hello")
        session.commit()

        store.delete_conversation(session, conv)
        session.commit()

        survivors = session.execute(
            select(QueryLog).where(QueryLog.conversation_id == conv)
        ).scalars().all()
        assert len(survivors) == 1
        # The dangling id is a correlation key that matches nothing now. Harmless,
        # and it keeps a deleted thread identifiable in the logs.
        assert survivors[0].original_query == "what is the refund window?"


class TestAppendTurn:
    def test_turn_index_is_monotonic(self, session: Session, conv: str):
        for i in range(4):
            store.append_turn(session, conv, role=TurnRole.USER, content=f"q{i}")
        session.commit()
        indexes = [t.turn_index for t in store.recent_turns(session, conv, limit=10)]
        assert indexes == [0, 1, 2, 3]

    def test_indexes_are_per_conversation(self, session: Session):
        a = store.create_conversation(session).conversation_id
        b = store.create_conversation(session).conversation_id
        store.append_turn(session, a, role=TurnRole.USER, content="a1")
        store.append_turn(session, b, role=TurnRole.USER, content="b1")
        store.append_turn(session, a, role=TurnRole.USER, content="a2")
        session.commit()

        assert [t.content for t in store.recent_turns(session, a, limit=10)] == ["a1", "a2"]
        assert [t.content for t in store.recent_turns(session, b, limit=10)] == ["b1"]

    def test_accepts_string_role(self, session: Session, conv: str):
        turn = store.append_turn(session, conv, role="assistant", content="hi")
        assert turn.role is TurnRole.ASSISTANT

    def test_citations_and_flags_default(self, session: Session, conv: str):
        turn = store.append_turn(session, conv, role=TurnRole.USER, content="q")
        assert turn.citations == []
        assert turn.abstained is False
        assert turn.query_id is None

    def test_assistant_turn_records_citations_and_abstention(self, session: Session, conv: str):
        turn = store.append_turn(
            session,
            conv,
            role=TurnRole.ASSISTANT,
            content="no answer",
            citations=["chunk-a", "chunk-b"],
            query_id="q-1",
            abstained=True,
        )
        assert turn.citations == ["chunk-a", "chunk-b"]
        assert turn.abstained is True
        assert turn.query_id == "q-1"

    def test_append_touches_conversation_recency(self, session: Session):
        """A conversation being used moves up the list (FR-25 sidebar)."""
        a = store.create_conversation(session).conversation_id
        b = store.create_conversation(session).conversation_id
        session.commit()

        before = [c.conversation_id for c in store.list_conversations(session)]
        assert before.index(b) < before.index(a), "b was created last, so it leads"

        store.append_turn(session, a, role=TurnRole.USER, content="hello")
        session.commit()

        after = [c.conversation_id for c in store.list_conversations(session)]
        assert after.index(a) < after.index(b), "activity on a must promote a"


class TestRecentTurns:
    def test_returns_oldest_first(self, session: Session, conv: str):
        for i in range(6):
            store.append_turn(session, conv, role=TurnRole.USER, content=f"turn-{i}")
        session.commit()

        recent = store.recent_turns(session, conv, limit=3)
        # Reversed order is the whole point: an LLM handed a reversed history
        # resolves anaphora against the wrong antecedent and does it confidently.
        assert [t.content for t in recent] == ["turn-3", "turn-4", "turn-5"]

    def test_limit_larger_than_thread_returns_all(self, session: Session, conv: str):
        store.append_turn(session, conv, role=TurnRole.USER, content="only")
        session.commit()
        assert len(store.recent_turns(session, conv, limit=100)) == 1

    def test_zero_limit_returns_nothing(self, session: Session, conv: str):
        store.append_turn(session, conv, role=TurnRole.USER, content="x")
        session.commit()
        assert store.recent_turns(session, conv, limit=0) == []

    def test_empty_conversation(self, session: Session, conv: str):
        assert store.recent_turns(session, conv, limit=5) == []


class TestHistoryViews:
    def test_message_history_interleaves_roles(self, session: Session, conv: str):
        store.append_turn(session, conv, role=TurnRole.USER, content="what about shipping?")
        store.append_turn(session, conv, role=TurnRole.ASSISTANT, content="5 business days.")
        session.commit()

        assert store.message_history(session, conv, limit=6) == [
            {"role": "user", "content": "what about shipping?"},
            {"role": "assistant", "content": "5 business days."},
        ]

    def test_query_history_excludes_assistant_turns(self, session: Session, conv: str):
        """The retriever must see questions only.

        `extract_topic` reduces a turn to a noun phrase by stripping interrogative
        leads, so a full assistant paragraph becomes a mangled antecedent and a
        confidently wrong standalone query.
        """
        store.append_turn(session, conv, role=TurnRole.USER, content="refund window")
        store.append_turn(
            session,
            conv,
            role=TurnRole.ASSISTANT,
            content="Customers may request a refund within 30 days.",
        )
        store.append_turn(session, conv, role=TurnRole.USER, content="and shipping?")
        session.commit()

        assert store.query_history(session, conv, limit=10) == [
            "refund window",
            "and shipping?",
        ]

    def test_both_views_respect_the_limit(self, session: Session, conv: str):
        for i in range(6):
            store.append_turn(session, conv, role=TurnRole.USER, content=f"q{i}")
            store.append_turn(session, conv, role=TurnRole.ASSISTANT, content=f"a{i}")
        session.commit()

        assert len(store.message_history(session, conv, limit=4)) == 4
        assert len(store.query_history(session, conv, limit=4)) == 4

    def test_query_history_zero_limit(self, session: Session, conv: str):
        store.append_turn(session, conv, role=TurnRole.USER, content="q")
        session.commit()
        assert store.query_history(session, conv, limit=0) == []


class TestListing:
    def test_most_recent_first(self, session: Session):
        ids = [store.create_conversation(session).conversation_id for _ in range(3)]
        session.commit()
        listed = [c.conversation_id for c in store.list_conversations(session)]
        assert listed == list(reversed(ids))

    def test_pagination(self, session: Session):
        for _ in range(5):
            store.create_conversation(session)
        session.commit()
        assert len(store.list_conversations(session, limit=2)) == 2
        assert len(store.list_conversations(session, limit=2, offset=4)) == 1
        assert len(store.list_conversations(session, limit=2, offset=99)) == 0

    def test_negative_limit_is_clamped(self, session: Session):
        store.create_conversation(session)
        session.commit()
        assert store.list_conversations(session, limit=-5) == []