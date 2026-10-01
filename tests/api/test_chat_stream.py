"""`POST /chat/stream`: SSE ordering, refusal, QueryLog (FR-19, FR-20, FR-28)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import QueryLog


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name = ""
        data: dict = {}
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name:
            events.append((name, data))
    return events


def test_stream_emits_sources_then_tokens_then_done(
    client: TestClient, live_doc, settings_env, monkeypatch
):
    # Retrieve regardless of the calibrated threshold: this test is about the
    # streaming contract, not about the threshold band.
    monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)

    response = client.post(
        "/chat/stream",
        json={"message": "What is the refund window for digital products?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert "token" in names

    sources = events[0][1]["sources"]
    assert sources, "a non-empty retrieved set must be shown before tokens"
    assert {"index", "chunk_id", "breadcrumb", "page"} <= set(sources[0])

    done = events[-1][1]
    assert done["abstained"] is False
    assert done["query_id"]
    assert done["ttft_ms"] is not None

    # No internal id in the generated text.
    answer = "".join(data["text"] for name, data in events if name == "token")
    for source in sources:
        assert source["chunk_id"] not in answer


def test_abstention_short_circuits_to_refusal_with_sources(
    client: TestClient, live_doc, settings_env, monkeypatch
):
    # Nothing can clear a threshold above the score ceiling: deterministic abstain.
    monkeypatch.setattr(settings_env, "retrieval_threshold", 1.1, raising=False)

    response = client.post("/chat/stream", json={"message": "What is the refund window?"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert "token" in names
    assert names[-1] == "done"

    done = events[-1][1]
    assert done["abstained"] is True
    # FR-20: a refusal still shows sources.
    assert events[0][1]["sources"]

    answer = "".join(data["text"] for name, data in events if name == "token")
    assert "couldn't find" in answer


def test_query_log_written(
    client: TestClient, live_doc, session: Session, settings_env, monkeypatch
):
    monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)
    client.post("/chat/stream", json={"message": "refund window digital product"})

    rows = session.query(QueryLog).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.original_query == "refund window digital product"
    assert row.model
    assert row.prompt_version == "chat.v1"
    assert row.total_ms is not None
    assert isinstance(row.scores, dict)
    assert isinstance(row.retrieved_ids, list)


def test_oversize_message_rejected(client: TestClient, settings_env, monkeypatch):
    monkeypatch.setattr(settings_env, "chat_max_query_chars", 10, raising=False)
    response = client.post("/chat/stream", json={"message": "x" * 200})
    assert response.status_code == 400
    assert "too long" in response.json()["detail"].lower()


def test_a_failed_log_write_does_not_truncate_a_delivered_answer(
    client: TestClient, live_doc, session: Session, settings_env, monkeypatch
):
    """A query-log failure must cost the log row, not the user's answer.

    `persist()` runs after the last token is flushed, so a database error there used
    to escape the `except AppError` handler as an unhandled ASGI exception: the
    client had the full answer and then lost the connection with no `done` and no
    `error` event. The 50-VU load sweep reproduced it against a locked SQLite file,
    which is the argument for testing it rather than reasoning about it.
    """
    monkeypatch.setattr(settings_env, "retrieval_threshold", 0.0, raising=False)

    import app.api.chat as chat_module

    def boom(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(chat_module, "record_query", boom)

    response = client.post(
        "/chat/stream",
        json={"message": "What is the refund window for digital products?"},
    )
    assert response.status_code == 200
    events = _parse_sse(response.text)
    names = [name for name, _ in events]

    # The answer still reached the user, and the stream still terminated properly.
    assert "token" in names
    assert names[-1] == "done", (
        f"stream did not terminate cleanly: {names}. A log-write failure must not "
        f"leave the user with a truncated answer."
    )
    assert "error" not in names, (
        "emitting an error after the answer was already delivered contradicts text "
        "the user has read"
    )

    # ...and the loss is visible rather than silent: no row, no query id, so FR-29
    # feedback has nothing to attach to for this turn.
    assert session.query(QueryLog).count() == 0
    assert events[-1][1]["query_id"] is None


@pytest.mark.parametrize(
    ("delivered", "suppress", "expect_raise"),
    [
        # Answer already read by the user: lose the log row, keep the answer.
        (True, False, False),
        # Cleanup during unwind: never raise, or the original error is replaced.
        (True, True, False),
        (False, True, False),
        # Nothing promised yet and not cleaning up: a real service error.
        (False, False, True),
    ],
)
def test_safe_persist_policy(delivered, suppress, expect_raise, session):
    """The recovery policy, checked directly on all four combinations.

    The interesting property is not any single branch but the *distinction*: a
    database error after the user has read the answer is a lost log row, while the
    same error during `finally` or before any output is a genuine failure. Getting
    this backwards is how a stream gets truncated, and how a provider error gets
    replaced by a misleading database one.
    """
    from app.api.chat import _safe_persist, _StreamState

    state = _StreamState(abstained=False)
    if delivered:
        state.ttft_ms = 42

    def failing() -> str:
        raise RuntimeError("database is locked")

    if expect_raise:
        with pytest.raises(RuntimeError, match="database is locked"):
            _safe_persist(failing, session, state, suppress=suppress)
    else:
        assert _safe_persist(failing, session, state, suppress=suppress) is None

    # The failure is never silent, whatever the policy: it is recorded on the state
    # for the access log and for tests.
    assert "database is locked" in (state.persist_error or "")
    assert state.persist_attempted is False, "the wrapper must not claim a write happened"


def test_safe_persist_returns_the_id_and_marks_recorded(session):
    """The success path is unchanged: the id comes back and `recorded` is set."""
    from app.api.chat import _safe_persist, _StreamState

    state = _StreamState(abstained=False)
    state.ttft_ms = 7

    assert _safe_persist(lambda: "q-123", session, state, suppress=False) == "q-123"
    assert state.persist_error is None
