"""Feedback endpoint tests (FR-29, NFR-5).

Feedback is only useful to the improvement loop if it is attached to the right row
and survives a round trip, so these tests go through the store to create a real
`QueryLog` rather than mocking one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import QueryLog


@pytest.fixture
def query_id(session: Session) -> str:
    """A real logged query, so `rowcount` and the PK lookup have something to match."""
    entry = QueryLog(
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
    session.add(entry)
    session.commit()
    return entry.query_id


class TestSetFeedback:
    def test_records_thumbs_up(self, client: TestClient, query_id: str):
        response = client.put("/feedback", json={"query_id": query_id, "value": "up"})
        assert response.status_code == 200
        assert response.json() == {"query_id": query_id, "value": "up"}

    def test_records_thumbs_down(self, client: TestClient, query_id: str):
        client.put("/feedback", json={"query_id": query_id, "value": "down"})
        assert client.get(f"/feedback/{query_id}").json()["value"] == "down"

    def test_persists_to_the_row(self, client: TestClient, query_id: str, session: Session):
        client.put("/feedback", json={"query_id": query_id, "value": "down"})
        session.expire_all()
        assert session.get(QueryLog, query_id).feedback == "down"

    def test_switching_replaces(self, client: TestClient, query_id: str):
        """Reconsidering is not a conflict, and a double-click is not an error."""
        client.put("/feedback", json={"query_id": query_id, "value": "up"})
        client.put("/feedback", json={"query_id": query_id, "value": "down"})
        assert client.get(f"/feedback/{query_id}").json()["value"] == "down"

    def test_repeating_is_idempotent(self, client: TestClient, query_id: str):
        for _ in range(3):
            assert client.put(
                "/feedback", json={"query_id": query_id, "value": "up"}
            ).status_code == 200

    def test_none_clears(self, client: TestClient, query_id: str, session: Session):
        client.put("/feedback", json={"query_id": query_id, "value": "up"})
        client.put("/feedback", json={"query_id": query_id, "value": "none"})
        session.expire_all()
        assert session.get(QueryLog, query_id).feedback is None
        assert client.get(f"/feedback/{query_id}").json()["value"] == "none"

    def test_unknown_query_returns_404(self, client: TestClient):
        response = client.put("/feedback", json={"query_id": "nope", "value": "up"})
        assert response.status_code == 404

    def test_invalid_value_rejected(self, client: TestClient, query_id: str):
        assert client.put(
            "/feedback", json={"query_id": query_id, "value": "sideways"}
        ).status_code == 422

    def test_missing_fields_rejected(self, client: TestClient, query_id: str):
        assert client.put("/feedback", json={"value": "up"}).status_code == 422
        assert client.put("/feedback", json={"query_id": query_id}).status_code == 422


class TestReadFeedback:
    def test_defaults_to_none(self, client: TestClient, query_id: str):
        assert client.get(f"/feedback/{query_id}").json()["value"] == "none"

    def test_round_trips(self, client: TestClient, query_id: str):
        client.put("/feedback", json={"query_id": query_id, "value": "up"})
        assert client.get(f"/feedback/{query_id}").json()["value"] == "up"

    def test_unknown_returns_404(self, client: TestClient):
        assert client.get("/feedback/nope").status_code == 404


class TestClearFeedback:
    def test_delete_clears(self, client: TestClient, query_id: str, session: Session):
        client.put("/feedback", json={"query_id": query_id, "value": "down"})
        assert client.delete(f"/feedback/{query_id}").status_code == 204
        session.expire_all()
        assert session.get(QueryLog, query_id).feedback is None

    def test_unknown_returns_404(self, client: TestClient):
        """A silent 204 would hide a wrong-id client bug."""
        assert client.delete("/feedback/nope").status_code == 404


class TestImprovementLoop:
    def test_down_votes_are_filterable(self, client: TestClient, session: Session):
        """The reason feedback lives on QueryLog (FR-29 + FR-28).

        "show me the answers users disliked" must be a filter on the row that also
        carries the per-stage scores. A separate feedback table would need the same
        join, so the column would buy nothing.
        """
        from sqlalchemy import select

        entries = [
            QueryLog(
                original_query=f"q{i}",
                rewritten_query=f"q{i}",
                retrieved_ids=[],
                scores={},
                threshold_applied=0.3,
                abstained=False,
                model="fake",
                prompt_version="chat.v1",
                tokens_in=1,
                tokens_out=1,
                citations_stripped=0,
            )
            for i in range(3)
        ]
        session.add_all(entries)
        session.commit()

        for entry in entries[:2]:
            client.put("/feedback", json={"query_id": entry.query_id, "value": "down"})
        client.put("/feedback", json={"query_id": entries[2].query_id, "value": "up"})
        session.expire_all()

        disliked = session.execute(
            select(QueryLog).where(QueryLog.feedback == "down")
        ).scalars().all()
        assert len(disliked) == 2
        # And the scores needed to classify each failure are on the same row.
        assert all(e.scores is not None for e in disliked)