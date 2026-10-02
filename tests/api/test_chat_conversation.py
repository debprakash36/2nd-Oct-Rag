"""Chat conversation integration tests (FR-23, FR-25, FR-29, FR-31, FR-34).

These drive the real streaming endpoint rather than the store, because the things
worth protecting here are about *wiring*: that history actually reaches the
retriever and the prompt, that both turns land in the thread in the right order, and
that a failure partway through still leaves a usable thread.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.guardrails import reset_rate_limiter
from app.db.models import QueryLog, Turn, TurnRole


@pytest.fixture(autouse=True)
def _clear_limiter():
    """The process-wide limiter would otherwise leak across tests.

    FR-31's limit is per client, and the test client has one address, so a suite that
    made more chat calls than the limit would start failing with 429 in a test that
    has nothing to do with rate limiting.
    """
    reset_rate_limiter()
    yield
    reset_rate_limiter()


def _ask(client: TestClient, message: str, conversation_id: str | None = None) -> str:
    payload: dict[str, str] = {"message": message}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200, response.text
    assert '"query_id"' in response.text
    # Extract the query_id from the done event so feedback can be attached.
    for line in response.text.splitlines():
        if line.startswith("data:"):
            import json

            data = json.loads(line[5:])
            if "query_id" in data:
                return str(data["query_id"])
    raise AssertionError("no done event with a query_id")


class TestTurnPersistence:
    def test_question_and_answer_both_persisted(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        _ask(client, "What is the refund window for digital products?", cid)
        session.expire_all()

        turns = session.query(Turn).filter_by(conversation_id=cid).order_by(Turn.turn_index).all()
        assert [t.role for t in turns] == [TurnRole.USER, TurnRole.ASSISTANT]
        assert turns[0].content == "What is the refund window for digital products?"
        assert turns[1].content

    def test_turn_indexes_are_sequential_across_requests(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        _ask(client, "first question about refunds", cid)
        _ask(client, "second question about refunds", cid)
        session.expire_all()

        turns = session.query(Turn).filter_by(conversation_id=cid).order_by(Turn.turn_index).all()
        assert [t.turn_index for t in turns] == [0, 1, 2, 3]
        assert [t.role for t in turns] == [
            TurnRole.USER,
            TurnRole.ASSISTANT,
            TurnRole.USER,
            TurnRole.ASSISTANT,
        ]

    def test_assistant_turn_cites_the_retrieved_chunks(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        _ask(client, "refund window digital product", cid)
        session.expire_all()

        assistant = (
            session.query(Turn)
            .filter_by(conversation_id=cid, role=TurnRole.ASSISTANT)
            .one()
        )
        assert assistant.citations, "an answer drawn from passages must cite them"

        # Every cited chunk must be a real chunk of a live document, or the sources
        # panel will offer a citation that 404s when click.
        cited = set(assistant.citations)
        assert cited

        from app.db.models import Chunk

        real = {row.chunk_id for row in session.query(Chunk).all()}
        assert cited <= real, "a cited chunk id that does not exist would 404 on click"

        # FR-30: the turn and the log row are written in one transaction.
        assert session.query(QueryLog).filter_by(conversation_id=cid).count() == 1

    def test_assistant_turn_carries_its_query_id(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        """A reloaded thread must be able to offer feedback on each answer (FR-29).

        The `done` event gives the client a query id, but a page reload reads the
        conversation instead. Without `query_id` on the turn there is nothing to attach
        a vote to and the thumbs never appear.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        query_id = _ask(client, "refund window digital product", cid)
        session.expire_all()

        assistant = (
            session.query(Turn)
            .filter_by(conversation_id=cid, role=TurnRole.ASSISTANT)
            .one()
        )
        assert assistant.query_id == query_id
        assert session.get(QueryLog, assistant.query_id) is not None

        # And it is visible through the API the UI actually reloads from.
        detail = client.get(f"/conversations/{cid}").json()
        reloaded = next(t for t in detail["turns"] if t["role"] == "assistant")
        assert reloaded["query_id"] == query_id

    def test_abstention_is_recorded_on_the_turn(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        """A refusal is a turn, and it is flagged so the UI can style it (FR-20)."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 1.1, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        _ask(client, "something unrelated to the corpus", cid)
        session.expire_all()

        assistant = (
            session.query(Turn)
            .filter_by(conversation_id=cid, role=TurnRole.ASSISTANT)
            .one()
        )
        assert assistant.abstained is True
        assert assistant.citations, "FR-20: a refusal still shows what was consulted"

    def test_query_log_links_to_conversation(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        query_id = _ask(client, "refund window digital product", cid)
        session.expire_all()

        row = session.get(QueryLog, query_id)
        assert row.conversation_id == cid

    def test_chat_without_conversation_id_still_works(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """The single-shot path from Phase 3 must not regress."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        query_id = _ask(client, "refund window digital product")
        assert query_id

    def test_unknown_conversation_id_does_not_error(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        """A stale id starts a fresh thread rather than rejecting the question.

        The client's view is out of date — the useful response is to answer, and the
        turns are simply not recorded anywhere rather than being orphaned.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        query_id = _ask(client, "refund window", "stale-conversation-id")

        session.expire_all()
        assert session.query(Turn).count() == 0
        assert session.get(QueryLog, query_id).original_query == "refund window"

    def test_deleted_conversation_still_answers(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        assert client.delete(f"/conversations/{cid}").status_code == 204

        query_id = _ask(client, "refund window", cid)
        session.expire_all()
        assert session.query(Turn).count() == 0
        assert session.get(QueryLog, query_id) is not None


class TestHistoryReachesThePipeline:
    def test_history_is_passed_to_the_model(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """Both roles must reach `build_messages` (FR-23).

        Asserted at the seam rather than on the prompt text, because whether a
        particular phrasing *uses* the history is a model-behaviour question. What
        the code owes is that the turns are present.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        _ask(client, "What is the refund window for digital products?", cid)

        captured: dict[str, object] = {}
        import app.generation.prompt as prompt_module

        original = prompt_module.build_messages

        def spy(question, passages, style=prompt_module.AnswerStyle.CONCISE, **kwargs):
            captured.update(kwargs)
            return original(question, passages, style, **kwargs)

        monkeypatch.setattr("app.api.chat.build_messages", spy)
        _ask(client, "how long do I have?", cid)

        history = captured.get("history")
        assert history, "follow-up must send prior turns"
        assert [t["role"] for t in history] == ["user", "assistant"]
        assert history[0]["content"] == "What is the refund window for digital products?"

    def test_history_is_passed_to_the_retriever(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """The retriever needs user questions only, and not the current one.

        The current question appearing in its own history would make every anaphora
        rewrite resolve against the question being asked, which is circular and
        confidently wrong.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        _ask(client, "What is the refund window for digital products?", cid)

        captured: dict[str, object] = {}
        import app.retrieval.retriever as retriever_module

        original = retriever_module.Retriever.retrieve

        def spy(self, query, **kwargs):
            captured.update(kwargs)
            return original(self, query, **kwargs)

        monkeypatch.setattr("app.api.chat.Retriever.retrieve", spy)
        _ask(client, "how long do I have?", cid)

        history = captured.get("history")
        assert history == ["What is the refund window for digital products?"]
        assert "how long do I have?" not in history

    def test_retrieval_memory_window_is_ten_user_questions(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """Retrieval sees the last 10 user questions, not the prompt window."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "retrieval_memory_turns", 10, raising=False)
        monkeypatch.setattr(settings_env, "chat_history_turns", 4, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        for i in range(12):
            _ask(client, f"question number {i} about refunds", cid)

        captured: dict[str, object] = {}
        import app.retrieval.retriever as retriever_module

        original = retriever_module.Retriever.retrieve

        def spy(self, query, **kwargs):
            captured.update(kwargs)
            return original(self, query, **kwargs)

        monkeypatch.setattr("app.api.chat.Retriever.retrieve", spy)
        _ask(client, "how long is that?", cid)

        history = captured["history"]
        assert len(history) == 10
        assert history[0].startswith("question number 2")
        assert history[-1].startswith("question number 11")
        assert "how long is that?" not in history

    def test_first_message_has_no_history(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]

        captured: dict[str, object] = {}
        import app.generation.prompt as prompt_module

        original = prompt_module.build_messages

        def spy(question, passages, style=prompt_module.AnswerStyle.CONCISE, **kwargs):
            captured.update(kwargs)
            return original(question, passages, style, **kwargs)

        monkeypatch.setattr("app.api.chat.build_messages", spy)
        _ask(client, "What is the refund window?", cid)

        assert not captured.get("history")

    def test_history_window_is_bounded(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """A long thread must not grow the prompt without limit (FR-23)."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "chat_history_turns", 4, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        for i in range(4):
            _ask(client, f"question number {i} about refunds", cid)

        captured: dict[str, object] = {}
        import app.generation.prompt as prompt_module

        original = prompt_module.build_messages

        def spy(question, passages, style=prompt_module.AnswerStyle.CONCISE, **kwargs):
            captured.update(kwargs)
            return original(question, passages, style, **kwargs)

        monkeypatch.setattr("app.api.chat.build_messages", spy)
        _ask(client, "and one more thing", cid)

        history = captured["history"]
        # Capped, and the window covers the *most recent* prior turns. Truncating
        # from the front instead (what `recent_turns` would do if it forgot to
        # reverse) keeps the oldest turns and drops the immediately preceding ones,
        # silently breaking every "how long is that?" that depends on the turn right
        # before it.
        assert len(history) == 4
        assert [t["role"] for t in history] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
        assert history[-2]["content"].startswith("question number 3")
        # Four asks produced eight turns; the window is the last four, so it starts
        # at question 2. Asserting the exact boundary is the point — an off-by-one
        # would silently shift the window while keeping the length correct.
        assert history[0]["content"].startswith("question number 2")

    def test_history_disabled_sends_none(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "chat_history_turns", 0, raising=False)
        cid = client.post("/conversations").json()["conversation_id"]
        _ask(client, "first question about refunds", cid)

        captured: dict[str, object] = {}
        import app.generation.prompt as prompt_module

        original = prompt_module.build_messages

        def spy(question, passages, style=prompt_module.AnswerStyle.CONCISE, **kwargs):
            captured.update(kwargs)
            return original(question, passages, style, **kwargs)

        monkeypatch.setattr("app.api.chat.build_messages", spy)
        _ask(client, "second question about refunds", cid)

        assert not captured.get("history")

        # The context window and persistence are separate concerns: with history off
        # the turns must still be recorded. Before this was fixed, the early return
        # for `chat_history_turns <= 0` skipped the user turn, leaving the thread with
        # an answer and no question above it.
        detail = client.get(f"/conversations/{cid}").json()
        roles = [t["role"] for t in detail["turns"]]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert detail["turns"][2]["content"] == "second question about refunds"

    def test_history_is_not_used_across_conversations(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """Two threads must not leak context into each other (FR-23 scope)."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        a = client.post("/conversations").json()["conversation_id"]
        b = client.post("/conversations").json()["conversation_id"]
        _ask(client, "refund window for digital products", a)

        captured: dict[str, object] = {}
        import app.generation.prompt as prompt_module

        original = prompt_module.build_messages

        def spy(question, passages, style=prompt_module.AnswerStyle.CONCISE, **kwargs):
            captured.update(kwargs)
            return original(question, passages, style, **kwargs)

        monkeypatch.setattr("app.api.chat.build_messages", spy)
        _ask(client, "shipping times", b)

        assert not captured.get("history")


class TestFeedbackFlow:
    def test_answer_id_can_be_thanked_or_rated(
        self, client: TestClient, live_doc, session: Session, settings_env, monkeypatch
    ):
        """End to end: stream an answer, then vote on the `query_id` it returned."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        query_id = _ask(client, "refund window digital product")

        assert client.put(
            "/feedback", json={"query_id": query_id, "value": "down"}
        ).status_code == 200
        session.expire_all()
        assert session.get(QueryLog, query_id).feedback == "down"


class TestGuardrailsOnTheChatPath:
    def test_oversize_rejected_before_any_provider_call(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """FR-34: the cap must precede embedding, or it costs money to enforce."""
        monkeypatch.setattr(settings_env, "chat_max_query_chars", 10, raising=False)

        # Assert on `stream`, not on the provider being constructed. FastAPI
        # resolves dependencies before the handler body runs, so the provider object
        # is always built; what FR-34 actually forbids is a billed generation call.
        called = False
        import app.providers.base as base_provider

        original_stream = base_provider.GenerationProvider.stream

        def spy(self, *args, **kwargs):
            nonlocal called
            called = True
            return original_stream(self, *args, **kwargs)

        monkeypatch.setattr(base_provider.GenerationProvider, "stream", spy)

        response = client.post("/chat/stream", json={"message": "x" * 200})
        assert response.status_code == 400
        assert "too long" in response.json()["detail"].lower()
        assert not called, "a rejected message must not reach the provider"

    def test_rate_limit_returns_429_with_retry_after(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """FR-31.

        `Retry-After` is what lets a client back off; a 429 without it invites the
        client to guess, and clients guess immediately.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "chat_rate_limit_requests", 2, raising=False)
        monkeypatch.setattr(settings_env, "chat_rate_limit_window_seconds", 60.0, raising=False)

        assert client.post("/chat/stream", json={"message": "refund window"}).status_code == 200
        assert client.post("/chat/stream", json={"message": "refund window"}).status_code == 200

        response = client.post("/chat/stream", json={"message": "refund window"})
        assert response.status_code == 429
        assert int(response.headers["Retry-After"]) > 0
        assert "too quickly" in response.json()["detail"].lower()

    def test_oversize_does_not_consume_a_rate_limit_slot(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """An oversized payload should not cost the caller one of their own slots.

        Otherwise a client that retries after a rejected too-long message locks
        itself out for the rest of the window.
        """
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "chat_max_query_chars", 10, raising=False)
        monkeypatch.setattr(settings_env, "chat_rate_limit_requests", 1, raising=False)

        # "refund window" is longer than the 10-char cap, so raise the cap after the
        # rejection to isolate the rate-limit slot rather than the size cap.
        assert client.post("/chat/stream", json={"message": "x" * 500}).status_code == 400
        monkeypatch.setattr(settings_env, "chat_max_query_chars", 4000, raising=False)
        assert client.post("/chat/stream", json={"message": "refund window"}).status_code == 200

    def test_rate_limit_can_be_disabled(
        self, client: TestClient, live_doc, settings_env, monkeypatch
    ):
        """A limit of 0 must mean 'no limit', not 'block everything'."""
        monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
        monkeypatch.setattr(settings_env, "chat_rate_limit_requests", 0, raising=False)
        for _ in range(5):
            assert client.post(
                "/chat/stream", json={"message": "refund window"}
            ).status_code == 200