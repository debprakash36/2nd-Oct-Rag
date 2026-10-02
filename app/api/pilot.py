"""Pilot measurement endpoints. These report UNMEASURED rather than inventing a pass.

The admin UI reads these. They do not run a threshold sweep (that is a long eval)
and they do not classify every failed query on GET (classification hits retrieval).
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Query
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.models import QueryLog
from app.db.session import get_db
from app.pilot.history import load_history, snapshot_summary
from app.pilot.metrics import DEFAULT_MIN_SAMPLES, build_metrics, collect_from_session

router = APIRouter(prefix="/admin/pilot", tags=["pilot"])


class ClassifyIn(BaseModel):
    query: str = Field(min_length=1, max_length=4000)


@router.get("/metrics")
def pilot_metrics(
    min_samples: int = Query(default=DEFAULT_MIN_SAMPLES, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    """PRD §8.2 traffic gate. Thin samples stay UNMEASURED."""
    raw = collect_from_session(session)
    metrics = build_metrics(raw, min_samples)
    earned = all(m.state == "pass" for m in metrics) and bool(metrics)
    return {
        "gate": "earned" if earned else "not_earned",
        "min_samples": min_samples,
        "metrics": [m.as_dict() for m in metrics],
        "raw": raw,
        "note": (
            "UNMEASURED is not a pass. The Phase 6 exit gate needs real traffic "
            "and at least one classified improvement-loop cycle."
        ),
    }


@router.get("/review")
def pilot_review(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    """Refused and down-voted questions, not yet classified.

    Classification hits the retriever, so it is on-demand (`POST /review/classify`)
    rather than implicit in this list.
    """
    refusals = case((QueryLog.abstained.is_(True), 1), else_=0)
    downs = case((QueryLog.feedback == "down", 1), else_=0)
    rows = session.execute(
        select(
            QueryLog.original_query,
            func.sum(refusals).label("refusals"),
            func.sum(downs).label("down_votes"),
            func.count().label("occurrences"),
        )
        .where(or_(QueryLog.abstained.is_(True), QueryLog.feedback == "down"))
        .group_by(QueryLog.original_query)
        .order_by(func.sum(refusals).desc(), func.count().desc())
        .limit(limit)
    ).all()
    items = [
        {
            "query": row.original_query,
            "why": "refused" if int(row.refusals or 0) else "down",
            "refusals": int(row.refusals or 0),
            "down_votes": int(row.down_votes or 0),
            "occurrences": int(row.occurrences or 0),
        }
        for row in rows
    ]
    return {
        "items": items,
        "empty_reason": None
        if items
        else (
            "The log has no refusals and no down-votes. Either the pilot has not "
            "run, or the system is answering everything — which PRD §8.2 calls "
            "the most damaging failure mode."
        ),
    }


@router.post("/review/classify")
def classify_query(
    body: ClassifyIn,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Run the improvement-loop classifier for one question."""
    from app.pilot.gaps import diagnose

    try:
        diagnosis = diagnose(session, body.query, settings)
    except Exception as exc:
        raise AppError(
            f"classification failed: {type(exc).__name__}",
            user_message="Could not classify that question. Try again.",
            status_code=500,
        ) from exc
    return diagnosis.as_dict()


@router.get("/threshold-history")
def threshold_history() -> dict[str, object]:
    """Recorded threshold sweeps. One snapshot is not a trend."""
    snapshots = [snapshot_summary(s) for s in load_history()]
    return {
        "snapshots": snapshots,
        "note": (
            "Activity 1 asks for a monthly re-sweep of real traffic. The eval-set "
            "distribution is not a substitute, and a single snapshot is not a trend."
        ),
    }
