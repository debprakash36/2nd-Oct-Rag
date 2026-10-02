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
import sys
from pathlib import Path

from app.pilot.metrics import (
    DEFAULT_MIN_SAMPLES,
    Counts,
    Metric,
    build_metrics,
    collect_from_sqlite,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Tests load this file by path and call `collect` / `build_metrics`.
collect = collect_from_sqlite


def render(metrics: list[Metric], data: Counts | dict[str, str]) -> None:
    print("PRD 8 pilot gate")
    print("=" * 68)
    for m in metrics:
        if m.value is None:
            value = "     -"
        elif m.value <= 1:
            value = f"{m.value:6.1%}"
        else:
            value = f"{m.value:6.0f}ms"
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

    return 1 if any(m.state == "fail" for m in metrics) else 0


if __name__ == "__main__":
    raise SystemExit(main())
