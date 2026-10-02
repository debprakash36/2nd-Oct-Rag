"""Phase 6 activity 1: re-sweep the threshold and **record the delta**.

## Why this exists

`implementation.md` §8.1(1): "The Phase 2 threshold was calibrated on 150-300
hand-picked questions. Real traffic has a different distribution. Re-sweep monthly
and **record the delta**."

`eval/run_eval.py --sweep` already produces the numbers. It prints them to stdout and
discards them, so the "record the delta" half of the instruction has nothing to record
*into*. Two monthly sweeps are then two unrelated terminal scrollbacks, and the one
question the phase is trying to answer — did this get better or worse, and because of
what — has no evidence behind it.

This appends each sweep to a JSONL history and reports the movement, so the answer is
"recall@10 fell 0.031 at t=0.10 after commit abc1234 with 297→312 chunks" rather than a
feeling.

## What a delta is allowed to mean

A number moving is not automatically a regression. Three things move it, and they have
different responses:

| Cause | Signal in the record | Response |
| --- | --- | --- |
| The **corpus** changed | `chunks` differs | Re-tune deliberately; a lower recall may be correct |
| The **code** changed | `commit` differs | Investigate — a retrieval change moved the number |
| **Neither** | both identical | The number moved for no traceable reason; treat it as noise |

Every snapshot records the commit and the chunk count precisely so the third case is
distinguishable from the first two. Without that, a delta is unattributable and the
phase degenerates into arguing about whether a number "really" changed.

## Usage

    python scripts/retune_threshold.py                    # sweep, record, show delta
    python scripts/retune_threshold.py --questions 60     # faster monthly run
    python scripts/retune_threshold.py --history         # show the trend, no sweep
    python scripts/retune_threshold.py --thresholds 0.05,0.10,0.15
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

#: One JSON object per line, appended. JSONL rather than a single JSON array so a
#: run is never at risk from an interrupted write corrupting the whole history, and
#: so it diffs cleanly in git -- which is where this record is most often read.
HISTORY_PATH = REPO_ROOT / "docs" / "eval" / "threshold_history.jsonl"

DEFAULT_THRESHOLDS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

#: A recall movement at or below this is treated as noise rather than a trend. The
#: eval set is 200 questions, so one question is worth 0.005 of recall@10; anything
#: under a few questions' worth should not be narrated as a regression.
NOISE_FLOOR = 0.01


@dataclass(frozen=True)
class Point:
    """One threshold's result."""

    threshold: float
    recall_at_10: float
    refusal_rate: float
    recall_ok: bool
    band_ok: bool

    @property
    def joint(self) -> bool:
        """Satisfies both targets at once, which is what §3.3 requires."""
        return self.recall_ok and self.band_ok

    def as_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "recall_at_10": round(self.recall_at_10, 4),
            "refusal_rate": round(self.refusal_rate, 4),
            "joint": self.joint,
        }


def dataset_size(path: Path, limit: int = 0) -> tuple[int, str]:
    """(questions actually evaluated, dataset fingerprint) for a dataset file.

    The count must reflect `--questions`, not the file length. `run_eval.py` slices
    the dataset before evaluating, so recording the full file would make a 40-question
    run indistinguishable from a 200-question one in the history -- and the
    comparability guard below would then wave through a delta it should have
    refused. That is not hypothetical: this function originally reported the file
    length, and the guard silently compared 40 questions against 200 and called it a
    regression.

    The fingerprint covers only the questions actually evaluated, for the same
    reason: a subset and the full set are different samples and must not hash alike.
    """
    if not path.exists():
        return 0, "missing"
    import hashlib

    questions: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                questions.append(json.loads(line).get("question", ""))
            except json.JSONDecodeError:
                continue
    if limit:
        questions = questions[:limit]
    digest = hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()[:12]
    return len(questions), digest


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=15
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _corpus_fingerprint(db_path: Path) -> dict[str, int]:
    """Chunk and document counts, so a delta can be attributed to the corpus."""
    import sqlite3

    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return {
            "chunks": connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "documents": connection.execute("SELECT count(*) FROM documents").fetchone()[0],
        }
    except sqlite3.Error:
        return {"chunks": 0, "documents": 0}
    finally:
        connection.close()


def sweep(thresholds: list[float], questions: int, dataset: Path) -> list[Point]:
    """Run the harness's own sweep and reduce it to comparable points.

    The measurement itself is *not* reimplemented. `sweep_threshold` and the two
    target constants live in `eval/run_eval.py`, and a second implementation here
    would drift from the one every other quality number comes from -- which is how a
    "regression" turns out to be two code paths measuring the same thing differently.
    """
    from app.core.config import get_settings
    from app.db.session import session_scope
    from app.retrieval.retriever import RetrievalConfig
    from eval.run_eval import (
        RECALL_AT_10_TARGET,
        REFUSAL_BAND,
        load_dataset,
        sweep_threshold,
    )

    settings = get_settings()
    rows = load_dataset(dataset)
    if questions:
        rows = rows[:questions]

    base = RetrievalConfig.from_settings(settings)
    with session_scope(settings) as session:
        reports = sweep_threshold(
            session,
            rows,
            base_config=base,
            settings=settings,
            thresholds=thresholds,
        )

    points: list[Point] = []
    for report in reports:
        # `label` is "threshold=0.10"; parsed rather than reformatted so the history
        # agrees with what `--sweep` printed.
        try:
            value = float(report.label.split("=")[-1])
        except (ValueError, IndexError):
            continue
        points.append(
            Point(
                threshold=value,
                recall_at_10=report.recall_at_10,
                refusal_rate=report.refusal_rate,
                recall_ok=report.recall_at_10 >= RECALL_AT_10_TARGET,
                band_ok=REFUSAL_BAND[0] <= report.refusal_rate <= REFUSAL_BAND[1],
            )
        )
    return sorted(points, key=lambda p: p.threshold)


def record(
    points: list[Point],
    db_path: Path,
    note: str,
    questions: int,
    fingerprint: str,
) -> dict[str, Any]:
    """Append one snapshot. Returns what was written."""
    from app.core.config import get_settings

    joint = [p for p in points if p.joint]
    snapshot: dict[str, Any] = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "corpus": _corpus_fingerprint(db_path),
        "embedding_provider": get_settings().embedding_provider,
        "embedding_model": get_settings().embedding_model,
        # Recorded so a delta between runs of different sizes can be refused rather
        # than reported. Recall@10 over 40 questions is not comparable to recall@10
        # over 200, and without this the tool will happily call the difference an
        # improvement.
        "questions": questions,
        "dataset_fingerprint": fingerprint,
        "note": note,
        "points": [p.as_dict() for p in points],
        # The recommended point is the highest-recall one that also satisfies the
        # refusal band. Recording it in the snapshot means a later run can say
        # "the recommendation moved from 0.10 to 0.15" without recomputing history.
        "recommended": max(joint, key=lambda p: p.recall_at_10).threshold if joint else None,
        "joint_count": len(joint),
    }
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot) + "\n")
    return snapshot


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in HISTORY_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line from an interrupted run is not worth
                # discarding the rest of the history over.
                continue
    return out


def show_delta(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """Movement, with the cause attributed where the record allows."""
    prev = {p["threshold"]: p for p in previous["points"]}
    curr = {p["threshold"]: p for p in current["points"]}

    corpus_changed = previous.get("corpus") != current.get("corpus")
    code_changed = previous.get("commit") != current.get("commit")

    print("\ndelta vs previous snapshot")
    print("=" * 72)
    print(
        f"  current   {current['recorded_at'][:19]}  commit {current['commit']}  "
        f"corpus {current['corpus']['chunks']} chunks  n={current.get('questions', '?')}"
    )
    print(
        f"  previous  {previous['recorded_at'][:19]}  commit {previous['commit']}  "
        f"corpus {previous['corpus']['chunks']} chunks  n={previous.get('questions', '?')}"
    )

    # A run over a different number of questions, or against edited questions, has
    # no comparable recall@10. Reporting the difference as an improvement is the
    # specific way a delta tool manufactures progress that never happened -- this
    # script did exactly that on its first run -- so the comparison is refused
    # outright rather than annotated and then summarised anyway.
    size_differs = previous.get("questions") != current.get("questions")
    set_differs = previous.get("dataset_fingerprint") != current.get(
        "dataset_fingerprint"
    )

    if size_differs or set_differs:
        print()
        if size_differs:
            print(
                f"  NOT LIKE-FOR-LIKE: sample size changed "
                f"({previous.get('questions')} -> {current.get('questions')} "
                f"questions). Recall@10 is not comparable across sizes, so no "
                f"movement is claimed."
            )
        if set_differs:
            print(
                "  NOT LIKE-FOR-LIKE: the dataset changed "
                f"({previous.get('dataset_fingerprint')} -> "
                f"{current.get('dataset_fingerprint')}). A delta against edited "
                f"questions measures the edit, not the system."
            )
        print("\n  current values, for the record:")
        print(f"  {'threshold':>9} {'recall@10':>10} {'refusal':>9}")
        for threshold in sorted(curr):
            print(
                f"  {threshold:>9.2f} {curr[threshold]['recall_at_10']:>10.3f} "
                f"{curr[threshold]['refusal_rate']:>9.1%}"
            )
        return

    print()
    print(f"  {'threshold':>9} {'recall@10':>18} {'refusal':>16}  change")
    for threshold in sorted(curr):
        now = curr[threshold]
        before = prev.get(threshold)
        if before is None:
            print(f"  {threshold:>9.2f} {now['recall_at_10']:>18.3f} "
                  f"{now['refusal_rate']:>16.1%}  new point")
            continue
        d_recall = now["recall_at_10"] - before["recall_at_10"]
        d_refusal = now["refusal_rate"] - before["refusal_rate"]
        if abs(d_recall) <= NOISE_FLOOR:
            change = "within noise"
        else:
            change = "IMPROVED" if d_recall > 0 else "REGRESSED"
        print(
            f"  {threshold:>9.2f} {now['recall_at_10']:>18.3f} "
            f"{now['refusal_rate']:>16.1%}  {change} "
            f"({d_recall:+.3f} recall, {d_refusal:+.1%} refusal)"
        )

    # The regression that matters: a threshold that used to satisfy both targets and
    # no longer does. Reported on its own because it is the finding, while the
    # per-point table is context.
    lost = [
        t
        for t, p in prev.items()
        if p["joint"] and t in curr and not curr[t]["joint"]
    ]
    if lost:
        print()
        print(f"  REGRESSION: threshold(s) {', '.join(f'{t:.2f}' for t in sorted(lost))} "
              f"no longer satisfy both targets.")
        if corpus_changed:
            print("    The corpus changed, so re-tune deliberately before assuming a "
                  "retrieval regression.")
        elif code_changed:
            print("    The corpus did not change, so this is a code regression.")
        else:
            print("    Neither corpus nor commit changed; the numbers moved for no "
                  "traceable reason.")

    if previous.get("recommended") != current.get("recommended"):
        print()
        print(
            f"  RECOMMENDATION MOVED: {previous.get('recommended')} -> "
            f"{current.get('recommended')}"
        )


def show_trend(history: list[dict[str, Any]]) -> None:
    if not history:
        print(f"no history yet at {HISTORY_PATH.relative_to(REPO_ROOT)}")
        return
    print(f"threshold history ({len(history)} run(s))")
    print("=" * 72)
    print(f"  {'recorded':<21} {'commit':<9} {'chunks':>7} {'rec@10':>8} {'refusal':>8}  rec")
    for snap in history:
        recommended = snap.get("recommended")
        point = next(
            (p for p in snap["points"] if p["threshold"] == recommended), None
        )
        recall = f"{point['recall_at_10']:.3f}" if point else "-"
        refusal = f"{point['refusal_rate']:.1%}" if point else "-"
        print(
            f"  {snap['recorded_at'][:19]:<21} {snap['commit']:<9} "
            f"{snap['corpus']['chunks']:>7} {recall:>8} {refusal:>8}  {recommended}"
        )
    if history[-1].get("recommended") is None:
        print()
        print("  The most recent run has NO threshold satisfying both targets. Per "
              "implementation.md 4.4 that is a conversation about the target pair, "
              "not a number to tune around.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--questions", type=int, default=0, help="limit dataset size")
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
        help="comma-separated thresholds to sweep",
    )
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--note", default="", help="why this run was done")
    parser.add_argument("--history", action="store_true", help="show trend and exit")
    parser.add_argument(
        "--no-record", action="store_true", help="sweep without appending to history"
    )
    args = parser.parse_args()

    if args.history:
        show_trend(load_history())
        return 0

    thresholds = [float(t) for t in args.thresholds.split(",") if t.strip()]
    print(f"sweeping {len(thresholds)} threshold(s) over {args.dataset}")
    points = sweep(thresholds, args.questions, REPO_ROOT / args.dataset)
    if not points:
        print("no results -- is the corpus empty?", file=sys.stderr)
        return 2

    print(f"\n{'threshold':>10} {'recall@10':>10} {'refusal':>9}  targets")
    for p in points:
        if p.joint:
            verdict = "both"
        elif p.recall_ok:
            verdict = "recall only"
        elif p.band_ok:
            verdict = "band only"
        else:
            verdict = "neither"
        print(
            f"{p.threshold:>10.2f} {p.recall_at_10:>10.3f} "
            f"{p.refusal_rate:>9.1%}  {verdict}"
        )

    joint = [p for p in points if p.joint]
    if not joint:
        print("\n  No threshold satisfies both targets simultaneously.")
        print("  implementation.md 4.4 is explicit that this is a PRD-level")
        print("  conversation, not something to tune around.")
    else:
        best = max(joint, key=lambda p: p.recall_at_10)
        print(f"\n  best: threshold={best.threshold:.2f} "
              f"recall@10={best.recall_at_10:.3f} refusal={best.refusal_rate:.1%}")

    if args.no_record:
        return 0

    from app.core.config import get_settings

    configured = get_settings().database_url
    if ":///" in configured:
        db_path = REPO_ROOT / configured.split(":///")[-1]
    else:
        db_path = REPO_ROOT / "rag.db"
    if not db_path.exists():
        print(
            f"  corpus {db_path.name} not found; recording with a zero fingerprint",
            file=sys.stderr,
        )
        db_path = REPO_ROOT / "rag.db"
    total, fingerprint = dataset_size(REPO_ROOT / args.dataset, args.questions)
    history = load_history()
    snapshot = record(points, db_path, args.note, total, fingerprint)
    print(f"\n  recorded to {HISTORY_PATH.relative_to(REPO_ROOT)}")

    if history:
        show_delta(history[-1], snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
