"""Admin API for Phase 6 measurement. UNMEASURED must survive the HTTP layer."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import QueryLog


def _log(
    session: Session,
    query: str,
    *,
    abstained: bool = False,
    feedback: str | None = None,
    ttft_ms: int | None = 800,
) -> None:
    session.add(
        QueryLog(
            original_query=query,
            rewritten_query=query,
            retrieved_ids=[],
            scores={},
            abstained=abstained,
            feedback=feedback,
            ttft_ms=ttft_ms,
            model="fake",
            prompt_version="chat.v1",
        )
    )
    session.commit()


def test_thin_traffic_is_unmeasured_not_a_pass(client: TestClient, session: Session) -> None:
    for _ in range(4):
        _log(session, "same question", feedback="up")

    body = client.get("/admin/pilot/metrics").json()
    assert body["gate"] == "not_earned"
    helpful = next(m for m in body["metrics"] if m["name"] == "answered-helpfully")
    assert helpful["state"] == "unmeasured"
    assert helpful["value"] == 1.0
    assert "UNMEASURED" in body["note"] or "not a pass" in body["note"].lower()


def test_review_lists_refusals_not_upvotes(client: TestClient, session: Session) -> None:
    _log(session, "refused question", abstained=True)
    _log(session, "liked question", feedback="up")

    body = client.get("/admin/pilot/review").json()
    queries = {item["query"] for item in body["items"]}
    assert "refused question" in queries
    assert "liked question" not in queries


def test_threshold_history_is_readable(client: TestClient) -> None:
    body = client.get("/admin/pilot/threshold-history").json()
    assert "snapshots" in body
    assert "trend" in body["note"].lower() or "monthly" in body["note"].lower()


def test_classify_requires_a_query(client: TestClient) -> None:
    response = client.post("/admin/pilot/review/classify", json={"query": ""})
    assert response.status_code == 422
