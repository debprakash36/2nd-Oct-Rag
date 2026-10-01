"""Conversations API tests (FR-25, FR-23, NFR-5)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.db.models import Turn, TurnRole


def _new(client: TestClient) -> str:
    response = client.post("/conversations")
    assert response.status_code == 201
    return response.json()["conversation_id"]


def _add_turn(client: TestClient, conversation_id: str, role: str, content: str) -> None:
    """Append a turn directly in the store, bypassing the streaming endpoint.

    Most of these tests are about the conversation API, and driving a full
    generation to obtain a turn would make them fail for retrieval reasons whenever
    the corpus or threshold changes.
    """
    from app.db import conversation as store
    from app.db.session import get_session_factory

    # Safe because the `client` fixture depends on `engine`, which registers the
    # test-scoped session factory. Without that, this no-argument call resolved to
    # the *configured* database and wrote rows into the real `rag.db`.
    session = get_session_factory()()
    try:
        store.append_turn(session, conversation_id, role=TurnRole(role), content=content)
        session.commit()
    finally:
        session.close()


class TestCreate:
    def test_returns_201_with_id(self, client: TestClient):
        response = client.post("/conversations")
        assert response.status_code == 201
        body = response.json()
        assert body["conversation_id"]
        assert body["turn_count"] == 0
        assert body["preview"] == "New conversation"

    def test_ids_are_distinct(self, client: TestClient):
        ids = {client.post("/conversations").json()["conversation_id"] for _ in range(5)}
        assert len(ids) == 5


class TestRead:
    def test_empty_conversation_roundtrips(self, client: TestClient):
        cid = _new(client)
        response = client.get(f"/conversations/{cid}")
        assert response.status_code == 200
        body = response.json()
        assert body["conversation_id"] == cid
        assert body["turns"] == []
        assert body["turn_count"] == 0

    def test_turns_are_chronological(self, client: TestClient):
        cid = _new(client)
        _add_turn(client, cid, "user", "first question")
        _add_turn(client, cid, "assistant", "first answer")
        _add_turn(client, cid, "user", "second question")

        turns = client.get(f"/conversations/{cid}").json()["turns"]
        assert [t["content"] for t in turns] == [
            "first question",
            "first answer",
            "second question",
        ]
        assert [t["turn_index"] for t in turns] == [0, 1, 2]
        assert [t["role"] for t in turns] == ["user", "assistant", "user"]

    def test_unknown_conversation_returns_404(self, client: TestClient):
        response = client.get("/conversations/does-not-exist")
        assert response.status_code == 404
        assert "Traceback" not in response.text

    def test_preview_uses_first_question(self, client: TestClient):
        cid = _new(client)
        _add_turn(client, cid, "user", "What is the refund window for digital products?")
        body = client.get(f"/conversations/{cid}").json()
        assert body["preview"] == "What is the refund window for digital products?"

    def test_preview_is_truncated_on_a_word_boundary(self, client: TestClient):
        cid = _new(client)
        long_question = "word " * 60
        _add_turn(client, cid, "user", long_question)
        preview = client.get(f"/conversations/{cid}").json()["preview"]
        assert len(preview) <= 73
        assert preview.endswith("…")
        assert "wor…" not in preview, "must not cut mid-word"

    def test_preview_collapses_newlines(self, client: TestClient):
        """A multi-line question must not break the sidebar layout."""
        cid = _new(client)
        _add_turn(client, cid, "user", "line one\n\nline two")
        assert client.get(f"/conversations/{cid}").json()["preview"] == "line one line two"


class TestList:
    def test_most_recent_first(self, client: TestClient):
        a, b, c = _new(client), _new(client), _new(client)
        listed = [row["conversation_id"] for row in client.get("/conversations").json()]
        assert listed == [c, b, a]

    def test_activity_reorders(self, client: TestClient):
        cid = _new(client)
        _add_turn(client, cid, "user", "hello")
        assert client.get("/conversations").json()[0]["conversation_id"] == cid

    def test_counts_and_previews(self, client: TestClient):
        cid = _new(client)
        _add_turn(client, cid, "user", "question one")
        _add_turn(client, cid, "assistant", "answer one")
        row = client.get("/conversations").json()[0]
        assert row["turn_count"] == 2
        assert row["preview"] == "question one"

    def test_pagination(self, client: TestClient):
        for _ in range(3):
            _new(client)
        assert len(client.get("/conversations", params={"limit": 2}).json()) == 2
        assert len(client.get("/conversations", params={"limit": 2, "offset": 2}).json()) == 1

    def test_limit_is_validated(self, client: TestClient):
        assert client.get("/conversations", params={"limit": 0}).status_code == 422
        assert client.get("/conversations", params={"limit": 999}).status_code == 422
        assert client.get("/conversations", params={"offset": -1}).status_code == 422


class TestDelete:
    def test_returns_204_and_hides_conversation(self, client: TestClient):
        cid = _new(client)
        _add_turn(client, cid, "user", "hello")
        assert client.delete(f"/conversations/{cid}").status_code == 204
        assert client.get(f"/conversations/{cid}").status_code == 404
        assert all(
            row["conversation_id"] != cid for row in client.get("/conversations").json()
        )

    def test_turns_are_removed(self, client: TestClient):
        from sqlalchemy import select

        from app.db.session import get_session_factory

        cid = _new(client)
        _add_turn(client, cid, "user", "hello")
        _add_turn(client, cid, "assistant", "hi")
        client.delete(f"/conversations/{cid}")

        session = get_session_factory()()
        try:
            assert (
                session.execute(select(Turn).where(Turn.conversation_id == cid))
                .scalars()
                .all()
                == []
            )
        finally:
            session.close()

    def test_unknown_conversation_returns_404(self, client: TestClient):
        assert client.delete("/conversations/does-not-exist").status_code == 404

    def test_deleting_twice_is_404_not_204(self, client: TestClient):
        """A silent 204 would hide a client bug behind a success."""
        cid = _new(client)
        assert client.delete(f"/conversations/{cid}").status_code == 204
        assert client.delete(f"/conversations/{cid}").status_code == 404