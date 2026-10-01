"""QueryLog write path (FR-28, architecture.md 5).

The value of this table is entirely in its completeness: a field that is not
persisted on the first write cannot be recovered later, and the classification the
PRD's improvement loop depends on (content gap vs. retrieval gap) needs the
per-stage scores and the rewritten query. The writer therefore takes everything
explicitly rather than deriving a convenient subset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import QueryLog
from app.retrieval.types import RetrievalCandidate, RetrievalResult

log = get_logger("app.db.query_log")


def scores_payload(candidates: Sequence[RetrievalCandidate]) -> dict[str, Any]:
    """Per-chunk, per-stage scores for the `scores` column.

    Shape: `{chunk_id: {"vector": 0.42, "keyword": 3.1, "fused": 0.03,
    "reranked": 0.71}}`. Only the stages that actually returned the candidate
    appear, because "absent" and "scored zero" are different facts and the
    distinction is what attributes a failure to a stage.
    """
    return {
        candidate.chunk_id: {
            stage.value: candidate.scores[stage] for stage in candidate.scores
        }
        for candidate in candidates
    }


def record_query(
    session: Session,
    *,
    original_query: str,
    result: RetrievalResult,
    retrieved: Sequence[RetrievalCandidate],
    model: str,
    prompt_version: str,
    tokens_in: int,
    tokens_out: int,
    ttft_ms: int | None,
    total_ms: int | None,
    citations_stripped: int,
    abstained: bool,
    conversation_id: str | None = None,
    trace_id: str | None = None,
) -> QueryLog:
    """Insert one QueryLog row and flush it so `query_id` is available.

    Flushed, not committed: the caller owns the transaction boundary. The
    streaming endpoint commits after the response completes, so a client that
    disconnects mid-answer is still recorded.
    """
    entry = QueryLog(
        trace_id=trace_id,
        conversation_id=conversation_id,
        original_query=original_query,
        rewritten_query=result.rewritten_query,
        retrieved_ids=[candidate.chunk_id for candidate in retrieved],
        scores=scores_payload(retrieved),
        threshold_applied=result.threshold_applied,
        abstained=abstained,
        model=model,
        prompt_version=prompt_version,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        citations_stripped=citations_stripped,
    )
    session.add(entry)
    session.flush()
    log.debug(
        "query logged",
        extra={
            "query_id": entry.query_id,
            "retrieved": len(retrieved),
            "abstained": abstained,
            "citations_stripped": citations_stripped,
        },
    )
    return entry
