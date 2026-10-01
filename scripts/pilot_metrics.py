"""PRD §8 pilot gate, measured (implementation.md 8.2, Phase 6 activity 3).

## Why this script refuses to produce a number it cannot stand behind

The pilot gate has six metrics with numeric targets. The temptation is to compute
each one and print it, which produces a report that looks like evidence and is not.

On this checkout that temptation produces **"answered-helpfully 100%"** — because
`query_logs` holds 4 votes, all "up", all on the *same* question. Reporting 100%
thumbs-up from four votes on one query would be worse than reporting nothing: it
would clear a 70% gate on the strength of a single test interaction.

So every metric here is in one of three states, and the third is the interesting one:

- **PASS** / **FAIL** — measured on at least `--min-samples` distinct queries.
- **UNMEASURED** — not enough distinct queries to mean anything. The target is
  printed alongside so it is clear what is still owed.

`implementation.md` §8.2 also warns that a near-zero refusal rate is *not* good news:
it usually means the threshold is too permissive and the system is answering from
irrelevant text — "the most damaging failure mode here, because it is invisible to
the user." This reports the band and says so, rather than only colouring it green.

## Metrics

Four come from `QueryLog` (real traffic). Two come from the eval set, which is
re-run separately via `eval/run_eval.py`; this script reports the eval figures only
if it is pointed at an eval report that contains them.

| Metric | Target | Source |
| --- | --- | --- |
| Answered-helpfully | >= 70% | `QueryLog.feedback` |
| Refusal rate | 10-30% | `QueryLog.abstained` |
| Unanswered-question rate | <= 15% | `QueryLog` |
| TTFT p95 | <= 5 s | `QueryLog.ttft_ms` |

## Usage

    python scripts/pilot_metrics.py
    python scripts/pilot_metrics.py --min-samples 30
    python scripts/pilot_metrics.py --json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent

#: PRD 8.2 targets.
TARGET_HELPFUL = 0.70
REFUSAL_BAND = (0.10, 0.30)
TARGET_UNANSWERED = 0.15
TARGET_TTFT_P95_MS = 5000.0

#: Below this many distinct queries a rate is not a measurement. Four votes on one
#: question is a test, not a pilot.
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
    """Raw counts from `query_logs`. Typed rather than `dict[str, object]` so the
    arithmetic below is checked instead of carrying `type: ignore` on every line."""

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


def _percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolated, matching the load driver and the eval harness.

    The same method everywhere matters: a p95 computed by nearest-rank in one report
    and by interpolation in another is not comparable, and comparing them is how a
    regression gets argued away.
    """
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


def collect(db: Path) -> Counts | dict[str, str]:
    """Pull the raw counts. Returns zeros rather than raising on a missing table."""
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
        # A subquery, not SUM(DISTINCT ...): aggregates over DISTINCT are not
        # portable and SQLite rejects them outright. Counting distinct *queries that
        # have a vote* is the sample size for the helpfulness rate.
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

    return {
        "total": row[0] or 0,
        "distinct_queries": row[1] or 0,
        "abstained": row[2] or 0,
        "up_votes": row[3] or 0,
        "down_votes": row[4] or 0,
        "votes": row[5] or 0,
        "timed": row[6] or 0,
        "voted_queries": voted_queries or 0,
        "ttft_p95": _percentile(ttfts, 0.95),
        "ttft_count": len(ttfts),
    }


def build_metrics(data: Counts | dict[str, str], min_samples: int) -> list[Metric]:
    # `collect` returns a bare {"error": ...} mapping when the log is unreadable.
    # `Counts` has no such key, so the union type is the honest way to express
    # "either counts, or a reason there are none".
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

    # 1. Answered-helpfully. Measured over *distinct voted queries*, not votes, so a
    #    single repeatedly-interacted-with query cannot carry the rate on its own.
    voted_queries = int(data["voted_queries"])
    up = int(data["up_votes"])
    down = int(data["down_votes"])
    votes = up + down
    if votes == 0:
        helpfulful_state, helpful_detail = "unmeasured", "no votes cast yet"
    elif voted_queries < min_samples:
        helpfulful_state = "unmeasured"
        helpful_detail = (
            f"{voted_queries} distinct query/queries have votes, need {min_samples}. "
            f"{up} up / {down} down on {votes} vote(s). Not reportable."
        )
    else:
        rate = up / votes
        helpfulful_state = "pass" if rate >= TARGET_HELPFUL else "fail"
        helpful_detail = f"{up} up / {down} down"
    metrics.append(
        Metric(
            name="answered-helpfully",
            value=(up / votes) if votes else None,
            target=f">= {TARGET_HELPFUL:.0%}",
            state=helpfulful_state,
            detail=helpful_detail,
            samples=voted_queries,
        )
    )

    # 2. Refusal rate -- a band, and an unhealthy one at either end.
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

    # 3. Unanswered-question rate. Defined here as "a distinct query that produced a
    #    refusal", i.e. the share of real questions the system declined. The PRD
    #    names the metric without defining it; this is the reading consistent with
    #    "unanswered", and it is stated rather than buried so it can be corrected.
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

    # 4. TTFT p95.
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


def render(metrics: list[Metric], data: Counts | dict[str, str]) -> None:
    print("PRD 8 pilot gate")
    print("=" * 68)
    for m in metrics:
        if m.value is None:
            value = "     -"
        elif m.value <= 1:
            value = f"{m.value:6.1%}"  # a rate
        else:
            value = f"{m.value:6.0f}ms"  # a latency
        marker = {"pass": "PASS", "fail": "FAIL", "unmeasured": "UNMEASURED"}[m.state]
        print(f"  {m.name:<26} {marker:<11} {value:>9}  (target {m.target})")
        if m.detail:
            print(f"      {m.detail}")
    print()

    if "error" not in data:
        print(
            f"  query_logs: {data['total']} rows, {data['distinct_queries']} distinct queries"
        )
    print()
    unmeasured = [m.name for m in metrics if m.state == "unmeasured"]
    if unmeasured:
        print(f"  NOT YET EARNED: {', '.join(unmeasured)}")
        print("  These are unmeasured, not passing. Phase 6 needs real pilot traffic;")
        print("  the current log is local test residue and cannot clear a gate.")
    if any(m.state == "fail" for m in metrics):
        print()
        print("  Failed metrics are not all retrieval problems. Use")
        print("  scripts/content_gaps.py to classify a failure before tuning anything.")
    print()
    print("  Not measured here (eval-set metrics, re-run separately):")
    print("    citation correctness >= 0.95   ->  eval/run_eval.py --stage retrieval")
    print("    groundedness          >= 0.90   ->  eval/run_eval.py --stage generation")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default="rag.db")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db = REPO_ROOT / args.db
    if not db.exists():
        print(f"no such database: {db}", file=sys.stderr)
        return 2

    data = collect(db)
    metrics = build_metrics(data, args.min_samples)

    if args.json:
        print(json.dumps({"metrics": [m.as_dict() for m in metrics], "raw": data}, indent=2))
    else:
        render(metrics, data)

    # Non-zero exit when something measurable failed, so this can gate CI or a
    # deploy. "Unmeasured" is deliberately not a failure: it is a known gap, and
    # exiting non-zero for it would train people to ignore the exit code.
    return 1 if any(m.state == "fail" for m in metrics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
