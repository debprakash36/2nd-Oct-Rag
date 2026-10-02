"""FR-30 / implementation.md 8.1(2): classify why a query failed, before acting on it.

## The problem this exists to solve

`implementation.md` §8.1 is blunt about it: a content gap and a retrieval gap need
*different* fixes, "classifying before acting is the entire point", and an agent that
"improves retrieval" for a content gap "will change nothing and report progress".

That failure mode is easy to fall into and invisible when it happens. Retuning a
threshold or adding keyword terms to a corpus that does not contain the answer
produces a clean git history, a passing eval, and zero change in user outcomes. So
this script does one thing: decide *which* of the problems you have.
## The taxonomy

| Classification | Evidence | Fix |
| --- | --- | --- |
| `content_gap` | Nothing in the corpus scores above noise, gate removed | Write the document |
| `threshold_gap` | Strong candidates exist; the gate abstained | Retune `retrieval_threshold` |
| `lexical_gap` | Vector half misses, keyword half finds it | Chunking or keyword index |
| `semantic_gap` | Keyword half misses, vector half finds it | Embedding model or chunk size |
| `ranking_gap` | Both halves find it; it missed the final context | Fusion, `rerank_k`, `top_k` |
| `answer_gap` | Retrieval was fine; the answer was rated down | Generation, not retrieval |

The diagnosis is differential: run the same query under the production config and
under progressively more permissive ones, and let the first config that *finds*
something name the layer that failed. Isolating the two halves of hybrid search
separately is what distinguishes `lexical_gap` from `semantic_gap` — a distinction
that matters because they have opposite fixes.

## Usage

    # Review the worst queries in the log (refusals + down-votes)
    python scripts/content_gaps.py

    # Ad-hoc: diagnose specific questions
    python scripts/content_gaps.py --query "What is the escalation policy?"
    python scripts/content_gaps.py --from eval/dataset.jsonl --limit 20

    # Machine-readable, for feeding a tuning loop
    python scripts/content_gaps.py --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.pilot.gaps import (  # noqa: E402
    Diagnosis,
    NOISE_FLOOR,
    STRONG_SIGNAL,
    diagnose,
    _best_score,
)


def queries_from_log(limit: int) -> list[tuple[str, str]]:
    """The queries a human should look at first: refused, then down-voted."""
    db = REPO_ROOT / "rag.db"
    if not db.exists():
        return []
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT original_query, "
            "       CASE WHEN abstained THEN 'refused' ELSE feedback END AS why "
            "FROM query_logs "
            "WHERE abstained = 1 OR feedback = 'down' "
            # One row per distinct question: a redelivered or retried query
            # otherwise fills the review list with copies of itself.
            "GROUP BY original_query "
            "ORDER BY SUM(abstained), COUNT(*) DESC "
            "LIMIT ?",
            (limit,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    return [(r[0], r[1]) for r in rows]


def queries_from_dataset(path: Path, limit: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            out.append((json.loads(line)["question"], "eval"))
            if len(out) >= limit:
                break
    return out


def render(diagnoses: Sequence[Diagnosis]) -> None:
    if not diagnoses:
        print("No queries to review.")
        print(
            "\n  The log has no refusals and no down-votes. Either the pilot has not\n"
            "  run, or the system is answering everything -- which PRD 8.2 calls the\n"
            "  most damaging failure mode, because a near-zero refusal rate means the\n"
            "  threshold is too permissive and the corpus does not cover the questions."
        )
        return

    for d in diagnoses:
        print(f"\n[{d.classification.upper()}]  ({d.confidence} confidence)")
        print(f"  query : {d.query}")
        print(f"  why   : {d.evidence}")
        print(f"  fix   : {d.fix}")
        for note in d.notes:
            print(f"  note  : {note}")

    counts: dict[str, int] = {}
    for d in diagnoses:
        counts[d.classification] = counts.get(d.classification, 0) + 1
    print("\n" + "=" * 68)
    print("summary")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16} {count}")
    print()
    print("  content_gap and answer_gap are NOT retrieval problems. Retuning the")
    print("  threshold for those changes nothing and will look like progress.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--query", action="append", help="diagnose this question")
    parser.add_argument("--from", dest="source", help="jsonl dataset of questions")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    settings = get_settings()

    if args.query:
        targets = [(q, "ad-hoc") for q in args.query]
    elif args.source:
        targets = queries_from_dataset(REPO_ROOT / args.source, args.limit)
    else:
        targets = queries_from_log(args.limit)

    if not targets:
        render([])
        return 0

    diagnoses: list[Diagnosis] = []
    with session_scope(settings) as session:
        for query, _why in targets:
            try:
                diagnoses.append(diagnose(session, query, settings))
            except Exception as exc:
                # Deliberately broad. One malformed query must not abort the review:
                # the list is worked through by hand, and losing 19 of 20 diagnoses
                # because of one bad row is worse than reporting the failure inline.
                # The classification is "error" so it cannot be mistaken for a verdict.
                diagnoses.append(
                    Diagnosis(
                        query=query, classification="error", confidence="high",
                        fix=f"diagnosis itself failed: {type(exc).__name__}: {exc}",
                        top_score=0.0, keyword_top=None, vector_top=None,
                        production_abstained=False, production_candidates=0,
                        threshold=settings.retrieval_threshold,
                    )
                )

    if args.json:
        print(json.dumps([d.as_dict() for d in diagnoses], indent=2))
    else:
        render(diagnoses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
