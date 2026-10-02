"""Classify a failed query before anyone retunes retrieval (implementation.md 8.1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import Document
from app.retrieval.retriever import RetrievalConfig, Retriever
from app.retrieval.types import RetrievalResult

NOISE_FLOOR = 1e-6
STRONG_SIGNAL = 0.05


@dataclass
class Diagnosis:
    query: str
    classification: str
    confidence: str
    fix: str
    top_score: float
    keyword_top: float | None
    vector_top: float | None
    production_abstained: bool
    production_candidates: int
    threshold: float
    evidence: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def best_score(result: object) -> float:
    scores: list[float] = []
    for candidate in getattr(result, "candidates", None) or []:
        try:
            scores.append(float(candidate.best_score()))
        except (AttributeError, TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def _count(result: object) -> int:
    return len(getattr(result, "candidates", []) or [])


# Script tests import `_best_score`. Keep that name as an alias.
_best_score = best_score


def diagnose(session: Session, query: str, settings: Settings) -> Diagnosis:
    """Classify one query by differencing production against more permissive configs."""
    base = RetrievalConfig.from_settings(settings)
    threshold = base.score_threshold

    def run(config: RetrievalConfig) -> RetrievalResult:
        return Retriever(session, settings=settings, config=config).retrieve(query)

    production = run(base)

    if not production.abstained:
        return Diagnosis(
            query=query,
            classification="answer_gap",
            confidence="high",
            fix=(
                "Retrieval returned candidates and the gate passed. The answer was "
                "still unhelpful, so this is generation or prompt, not retrieval. "
                "Do NOT retune the threshold for this query."
            ),
            top_score=best_score(production),
            keyword_top=None,
            vector_top=None,
            production_abstained=False,
            production_candidates=_count(production),
            threshold=threshold,
            evidence=f"{_count(production)} candidate(s) passed at threshold {threshold}",
        )

    permissive = replace(base, score_threshold=0.0, fetch_k=100, top_k=50)
    open_result = run(permissive)
    open_top = best_score(open_result)

    if open_top <= NOISE_FLOOR:
        live_docs = session.query(Document).count()
        return Diagnosis(
            query=query,
            classification="content_gap",
            confidence="high",
            fix=(
                "Nothing in the corpus scores above noise even with the threshold "
                f"removed, across {live_docs} live document(s). Retuning retrieval "
                "cannot fix a missing document. Write it."
            ),
            top_score=open_top,
            keyword_top=None,
            vector_top=None,
            production_abstained=True,
            production_candidates=0,
            threshold=threshold,
            evidence=f"best score {open_top:.6f} with the gate removed",
        )

    keyword_only = run(replace(permissive, vector_enabled=False))
    vector_only = run(replace(permissive, keyword_enabled=False))
    keyword_top = best_score(keyword_only)
    vector_top = best_score(vector_only)

    if keyword_top > NOISE_FLOOR and vector_top <= NOISE_FLOOR:
        return Diagnosis(
            query=query,
            classification="lexical_gap",
            confidence="high",
            fix=(
                "The keyword half finds this text and the vector half does not. The "
                "content is embedded unhelpfully -- look at chunk granularity, or "
                "whether the chunk boundary splits the answer across two chunks."
            ),
            top_score=open_top,
            keyword_top=keyword_top,
            vector_top=vector_top,
            production_abstained=True,
            production_candidates=0,
            threshold=threshold,
            evidence=f"keyword {keyword_top:.4f} vs vector {vector_top:.6f}",
        )

    if vector_top > NOISE_FLOOR and keyword_top <= NOISE_FLOOR:
        return Diagnosis(
            query=query,
            classification="semantic_gap",
            confidence="high",
            fix=(
                "The vector half finds this text and the keyword half does not. "
                "Check the keyword index and the glossary for the domain terms the "
                "question uses."
            ),
            top_score=open_top,
            keyword_top=keyword_top,
            vector_top=vector_top,
            production_abstained=True,
            production_candidates=0,
            threshold=threshold,
            evidence=f"vector {vector_top:.4f} vs keyword {keyword_top:.6f}",
        )

    if open_top >= STRONG_SIGNAL:
        return Diagnosis(
            query=query,
            classification="threshold_gap",
            confidence="high",
            fix=(
                f"Strong candidates (best {open_top:.4f}) exist but the gate at "
                f"{threshold} abstained. This is a threshold calibration problem. "
                "Re-sweep before changing it -- see eval/run_eval.py --sweep."
            ),
            top_score=open_top,
            keyword_top=keyword_top,
            vector_top=vector_top,
            production_abstained=True,
            production_candidates=0,
            threshold=threshold,
            evidence=f"best {open_top:.4f} with the gate removed vs threshold {threshold}",
        )

    return Diagnosis(
        query=query,
        classification="ranking_gap",
        confidence="low",
        fix=(
            f"Only weak signal is available (best {open_top:.4f}) and it did not "
            "survive fusion and reranking. Check fetch_k, rerank_k, and the fusion "
            "weights. Treat this classification as tentative: the corpus may simply "
            "not cover the question."
        ),
        top_score=open_top,
        keyword_top=keyword_top,
        vector_top=vector_top,
        production_abstained=True,
        production_candidates=0,
        threshold=threshold,
        evidence=f"best {open_top:.6f} with the gate removed",
        notes=[
            "low confidence: a content gap and a ranking gap look identical when "
            "the only available signal is weak"
        ],
    )
