"""PRD §8.2 traffic metrics. UNMEASURED is a first-class result, not a pass.

The CLI (`scripts/pilot_metrics.py`) and the admin API share this module so a
Postgres deployment and a local SQLite file cannot disagree on what a gate
clearance means.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

TARGET_HELPFUL = 0.70
REFUSAL_BAND = (0.10, 0.30)
TARGET_UNANSWERED = 0.15
TARGET_TTFT_P95_MS = 5000.0
DEFAULT_MIN_SAMPLES = 20


@dataclass
class Metric:
    name: str
    value: float | None
    target: str
    state: str  # "pass" | "fail" | "unmeasured"
    detail: str = ""
    samples: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class Counts(TypedDict):
    total: int
    distinct_queries: int
    abstained: int
    up_votes: int
    down_votes: int
    votes: int
    timed: int
    voted_queries: int
    ttft_p95: float
    ttft_count: int


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated, matching the load driver and the eval harness."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (1 - (position - lower)) + ordered[upper] * (position - lower)


def _counts_from_row(
    *,
    total: int,
    distinct_queries: int,
    abstained: int,
    up_votes: int,
    down_votes: int,
    votes: int,
    timed: int,
    voted_queries: int,
    ttfts: list[float],
) -> Counts:
    return {
        "total": total,
        "distinct_queries": distinct_queries,
        "abstained": abstained,
        "up_votes": up_votes,
        "down_votes": down_votes,
        "votes": votes,
        "timed": timed,
        "voted_queries": voted_queries,
        "ttft_p95": percentile(ttfts, 0.95),
        "ttft_count": len(ttfts),
    }


def collect_from_sqlite(db: Path) -> Counts | dict[str, str]:
    """Read `query_logs` from a SQLite file. Used by the CLI and its tests."""
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT "
            "  count(*) AS total, "
            "  count(DISTINCT original_query) AS distinct_queries, "
            "  sum(abstained) AS abstained, "
            "  sum(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END) AS up_votes, "
            "  sum(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END) AS down_votes, "
            "  sum(CASE WHEN feedback IN ('up','down') THEN 1 ELSE 0 END) AS votes, "
            "  sum(CASE WHEN ttft_ms IS NOT NULL THEN 1 ELSE 0 END) AS timed "
            "FROM query_logs"
        ).fetchone()
        voted_queries = connection.execute(
            "SELECT count(DISTINCT original_query) FROM query_logs "
            "WHERE feedback IN ('up','down')"
        ).fetchone()[0]
        ttfts = [
            float(r[0])
            for r in connection.execute(
                "SELECT ttft_ms FROM query_logs WHERE ttft_ms IS NOT NULL"
            )
        ]
    except sqlite3.Error as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        connection.close()

    return _counts_from_row(
        total=row[0] or 0,
        distinct_queries=row[1] or 0,
        abstained=row[2] or 0,
        up_votes=row[3] or 0,
        down_votes=row[4] or 0,
        votes=row[5] or 0,
        timed=row[6] or 0,
        voted_queries=voted_queries or 0,
        ttfts=ttfts,
    )


def collect_from_session(session: Session) -> Counts:
    """Same arithmetic as the SQLite collector, against whatever the app is serving."""
    from app.db.models import QueryLog

    abstained_expr = case((QueryLog.abstained.is_(True), 1), else_=0)
    up_expr = case((QueryLog.feedback == "up", 1), else_=0)
    down_expr = case((QueryLog.feedback == "down", 1), else_=0)
    vote_expr = case((QueryLog.feedback.in_(("up", "down")), 1), else_=0)
    timed_expr = case((QueryLog.ttft_ms.is_not(None), 1), else_=0)

    row = session.execute(
        select(
            func.count().label("total"),
            func.count(func.distinct(QueryLog.original_query)).label("distinct_queries"),
            func.coalesce(func.sum(abstained_expr), 0).label("abstained"),
            func.coalesce(func.sum(up_expr), 0).label("up_votes"),
            func.coalesce(func.sum(down_expr), 0).label("down_votes"),
            func.coalesce(func.sum(vote_expr), 0).label("votes"),
            func.coalesce(func.sum(timed_expr), 0).label("timed"),
        ).select_from(QueryLog)
    ).one()
    voted_queries = session.scalar(
        select(func.count(func.distinct(QueryLog.original_query))).where(
            QueryLog.feedback.in_(("up", "down"))
        )
    ) or 0
    ttfts = [
        float(ms)
        for ms in session.scalars(select(QueryLog.ttft_ms).where(QueryLog.ttft_ms.is_not(None)))
        if ms is not None
    ]
    return _counts_from_row(
        total=int(row.total or 0),
        distinct_queries=int(row.distinct_queries or 0),
        abstained=int(row.abstained or 0),
        up_votes=int(row.up_votes or 0),
        down_votes=int(row.down_votes or 0),
        votes=int(row.votes or 0),
        timed=int(row.timed or 0),
        voted_queries=int(voted_queries),
        ttfts=ttfts,
    )


# CLI name. Tests import `collect` from the script, which re-exports this.
collect = collect_from_sqlite


def build_metrics(data: Counts | dict[str, str], min_samples: int) -> list[Metric]:
    if "error" in data:
        return [
            Metric(
                name="query log",
                value=None,
                target="-",
                state="unmeasured",
                detail=f"could not read query_logs: {data.get('error', 'unknown')}",
            )
        ]

    distinct = int(data["distinct_queries"])
    metrics: list[Metric] = []

    voted_queries = int(data["voted_queries"])
    up = int(data["up_votes"])
    down = int(data["down_votes"])
    votes = up + down
    if votes == 0:
        helpful_state, helpful_detail = "unmeasured", "no votes cast yet"
    elif voted_queries < min_samples:
        helpful_state = "unmeasured"
        helpful_detail = (
            f"{voted_queries} distinct query/queries have votes, need {min_samples}. "
            f"{up} up / {down} down on {votes} vote(s). Not reportable."
        )
    else:
        rate = up / votes
        helpful_state = "pass" if rate >= TARGET_HELPFUL else "fail"
        helpful_detail = f"{up} up / {down} down"
    metrics.append(
        Metric(
            name="answered-helpfully",
            value=(up / votes) if votes else None,
            target=f">= {TARGET_HELPFUL:.0%}",
            state=helpful_state,
            detail=helpful_detail,
            samples=voted_queries,
        )
    )

    if distinct < min_samples:
        metrics.append(
            Metric(
                name="refusal rate",
                value=None,
                target=f"{REFUSAL_BAND[0]:.0%}-{REFUSAL_BAND[1]:.0%}",
                state="unmeasured",
                detail=f"{distinct} distinct queries, need {min_samples}",
                samples=distinct,
            )
        )
    else:
        rate = int(data["abstained"]) / distinct
        in_band = REFUSAL_BAND[0] <= rate <= REFUSAL_BAND[1]
        note = ""
        if rate < REFUSAL_BAND[0]:
            note = (
                "  Below the band usually means the threshold is too permissive and "
                "the system is answering from irrelevant text (PRD 8.2)."
            )
        elif rate > REFUSAL_BAND[1]:
            note = "  Above the band usually means the threshold is too strict."
        metrics.append(
            Metric(
                name="refusal rate",
                value=rate,
                target=f"{REFUSAL_BAND[0]:.0%}-{REFUSAL_BAND[1]:.0%}",
                state="pass" if in_band else "fail",
                detail=f"{int(data['abstained'])} of {distinct} abstained.{note}",
                samples=distinct,
            )
        )

    if distinct < min_samples:
        metrics.append(
            Metric(
                name="unanswered-question rate",
                value=None,
                target=f"<= {TARGET_UNANSWERED:.0%}",
                state="unmeasured",
                detail=f"{distinct} distinct queries, need {min_samples}",
                samples=distinct,
            )
        )
    else:
        rate = int(data["abstained"]) / distinct
        metrics.append(
            Metric(
                name="unanswered-question rate",
                value=rate,
                target=f"<= {TARGET_UNANSWERED:.0%}",
                state="pass" if rate <= TARGET_UNANSWERED else "fail",
                detail=(
                    "defined as refusals / distinct queries; the PRD does not define it"
                ),
                samples=distinct,
            )
        )

    timed = int(data["ttft_count"])
    if timed < min_samples:
        metrics.append(
            Metric(
                name="TTFT p95",
                value=None,
                target=f"<= {TARGET_TTFT_P95_MS:.0f} ms",
                state="unmeasured",
                detail=f"{timed} timed query/queries, need {min_samples}",
                samples=timed,
            )
        )
    else:
        p95 = float(data["ttft_p95"])
        metrics.append(
            Metric(
                name="TTFT p95",
                value=p95,
                target=f"<= {TARGET_TTFT_P95_MS:.0f} ms",
                state="pass" if p95 <= TARGET_TTFT_P95_MS else "fail",
                detail=f"from {timed} timed queries",
                samples=timed,
            )
        )

    return metrics
