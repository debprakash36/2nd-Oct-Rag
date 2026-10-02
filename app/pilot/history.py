"""Read the threshold-sweep JSONL without running a new sweep."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
HISTORY_PATH = REPO_ROOT / "docs" / "eval" / "threshold_history.jsonl"


def load_history(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or HISTORY_PATH
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def snapshot_summary(snap: dict[str, Any]) -> dict[str, Any]:
    recommended = snap.get("recommended")
    point = next(
        (p for p in snap.get("points") or [] if p.get("threshold") == recommended),
        None,
    )
    corpus = snap.get("corpus") or {}
    return {
        "recorded_at": snap.get("recorded_at"),
        "commit": snap.get("commit"),
        "chunks": corpus.get("chunks"),
        "documents": corpus.get("documents"),
        "embedding_model": snap.get("embedding_model"),
        "recommended": recommended,
        "recall_at_10": None if point is None else point.get("recall_at_10"),
        "refusal_rate": None if point is None else point.get("refusal_rate"),
        "note": snap.get("note") or "",
        "comparable": True,
    }
