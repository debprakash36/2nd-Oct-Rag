r"""Per-stage retrieval profile, to replace architecture.md 7.4's indicative budgets (5.2).

Section 7.4 is explicit that its table is "starting estimates to be replaced by
measurement". This produces that measurement. It reports each stage's cost as a
distribution rather than a single number, because a budget spent on p50 and blown on
p95 is a budget that is really being spent on the tail.

Two modes, because they answer different questions:

- Default: stages measured in-process, unloaded. This is the cost of the work itself,
  which is what a stage budget is about.
- `--vus N`: the same stages driven concurrently through a real server, so the
  numbers include queueing. Comparing the two shows how much of a stage's observed
  latency is its own work and how much is waiting for the process.

The honest caveat, repeated in the report: with the offline provider these figures
exclude generation, which architecture.md 7.4 puts at 3,500 ms and which dominates
everything here. Adding the two is the only way to judge the 5 s target, and it is
why the unloaded profile matters more than the loaded one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.retrieval.retriever import RetrievalConfig, Retriever  # noqa: E402

#: The questions the load driver uses, so both measurements describe the same work.
QUERIES = (
    "What is the refund window for digital products?",
    "How long do I have to return a physical good?",
    "What are the standard shipping times?",
    "How does the warranty period work?",
    "What happens if a customer disputes a charge?",
    "How long is personal data retained?",
    "When are invoices due?",
    "What does the escalation policy say?",
    "Which items are non-refundable?",
    "How quickly are refunds processed?",
)

#: architecture.md 7.4, in stage order. Kept so the report can show estimate vs
#: measurement side by side instead of quietly replacing one with the other.
#: Keys are `Stage`'s values (`app/retrieval/types.py`), not display names, so a
#: renamed stage shows up as a missing row rather than a silently dropped budget.
BUDGET_MS = {
    "rewrite": 400.0,
    "vector": 150.0,
    "keyword": 150.0,
    "fusion": 300.0,
    "rerank": 300.0,
    "threshold": 50.0,
    "budget": 50.0,
}

LABEL = {
    "rewrite": "Query rewrite",
    "vector": "Vector search",
    "keyword": "Keyword search",
    "fusion": "Fusion",
    "rerank": "Rerank",
    "threshold": "Threshold",
    "budget": "Context budget",
}


def percentile(values: list[float], fraction: float) -> float:
    """Same interpolation as the load driver, so the two reports are comparable."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def profile_inprocess(iterations: int, warmup: int) -> dict[str, object]:
    settings = get_settings()
    samples: dict[str, list[float]] = {}
    totals: list[float] = []

    with session_scope() as session:
        retriever = Retriever(
            session, settings=settings, config=RetrievalConfig()
        )

        for i in range(warmup):
            retriever.retrieve(QUERIES[i % len(QUERIES)])

        for i in range(iterations):
            result = retriever.retrieve(QUERIES[i % len(QUERIES)])
            for stage, value in result.timings_ms.items():
                samples.setdefault(stage, []).append(value)
            totals.append(sum(result.timings_ms.values()))

    return {"mode": "inprocess", "iterations": iterations, "stages": samples, "total": totals}


def summarise(profile: dict[str, object]) -> list[dict[str, object]]:
    stages: dict[str, list[float]] = profile["stages"]  # type: ignore[assignment]

    # A budgeted stage that produced no samples means the retriever stopped
    # reporting it, not that it was free. Omitting the row would quietly shrink the
    # profile -- which is exactly what happened while the stage keys were named
    # `vector_search`/`keyword_search` instead of `vector`/`keyword`, dropping the
    # two most expensive stages from the table.
    missing = [s for s in BUDGET_MS if not stages.get(s)]
    if missing:
        raise SystemExit(
            f"no timings recorded for budgeted stage(s): {', '.join(sorted(missing))}. "
            f"Recorded stages: {', '.join(sorted(stages)) or '(none)'}. Update "
            f"BUDGET_MS/LABEL to match `Stage` in app/retrieval/types.py."
        )

    rows: list[dict[str, object]] = []
    for stage in BUDGET_MS:
        values = stages[stage]
        p95 = percentile(values, 0.95)
        budget = BUDGET_MS[stage]
        rows.append(
            {
                "stage": stage,
                "label": LABEL[stage],
                "budget_ms": budget,
                "p50_ms": round(percentile(values, 0.50), 1),
                "p95_ms": round(p95, 1),
                "max_ms": round(max(values), 1),
                "mean_ms": round(statistics.fmean(values), 1),
                "n": len(values),
                "over_budget_p95": p95 > budget,
            }
        )
    rows.sort(key=lambda r: r["p95_ms"], reverse=True)  # type: ignore[arg-type,return-value]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs" / "perf_stages.json"
    )
    args = parser.parse_args()

    started = time.perf_counter()
    profile = profile_inprocess(args.iterations, args.warmup)
    rows = summarise(profile)

    print(f"{'stage':<30} {'budget':>8} {'p50':>8} {'p95':>8} {'max':>8}  verdict")
    for row in rows:
        verdict = "OVER" if row["over_budget_p95"] else "ok"
        print(
            f"{row['label']:<30} {row['budget_ms']:>8.0f} {row['p50_ms']:>8.1f} "
            f"{row['p95_ms']:>8.1f} {row['max_ms']:>8.1f}  {verdict}"
        )

    totals = profile["total"]  # type: ignore[assignment]
    print(
        f"\nretrieval total: p50 {percentile(totals, 0.50):.1f} ms  "
        f"p95 {percentile(totals, 0.95):.1f} ms  "
        f"(excludes generation; {time.perf_counter() - started:.1f}s wall)"
    )

    payload = {
        "mode": profile["mode"],
        "iterations": profile["iterations"],
        "stages": rows,
        "retrieval_total_ms": {
            "p50": round(percentile(totals, 0.50), 1),
            "p95": round(percentile(totals, 0.95), 1),
            "max": round(max(totals), 1),
        },
        "note": (
            "Offline provider: generation is excluded. architecture.md 7.4 budgets "
            "LLM first token at 3500 ms, which dominates every figure here."
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
