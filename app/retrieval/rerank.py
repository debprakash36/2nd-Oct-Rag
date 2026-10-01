"""Stage 3 of retrieval: cross-encoder rerank (architecture.md §3.3, FR-13).

architecture.md §3.3 calls reranking the single largest quality-per-unit-of-effort
lever in the system, and the reason it can afford to exist is the over-fetch that
happens before it: wide fetch returns 20-50 candidates and the narrow context
keeps 4-8, so the reranker has room to reorder rather than merely filter.

This module owns the stage logic and nothing else. The provider interface lives in
`app/providers/base.py` (NFR-10), so swapping a cross-encoder for a lexical
stand-in does not touch this file — which is what makes §10 check 2's A/B
possible.

## The local reranker is not a cross-encoder

`LexicalRerankerProvider` in `app/providers/rerank.py` is a deterministic
overlap scorer, not a trained cross-encoder. It exists because this environment
has no network access to a model host, and a phase gate that can only be measured
by hand-waving the rerank stage is not a measurement.

Consequence, stated plainly because it changes what the eval numbers mean: rerank
quality measured with this provider measures *the pipeline*, not *a cross-encoder*.
§10 check 2 can be run, and the numbers are real numbers, but they cannot
establish that a real cross-encoder earns its latency. Re-run that check against
a real provider before treating it as settled. `docs/eval_results.md` repeats
this caveat next to the measurement rather than burying it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.providers.base import RerankerProvider
from app.retrieval.types import RetrievalCandidate, Stage

log = logging.getLogger(__name__)


def rerank_candidates(
    query: str,
    candidates: Sequence[RetrievalCandidate],
    reranker: RerankerProvider,
    *,
    top_k: int,
) -> list[RetrievalCandidate]:
    """Score each candidate against `query` and re-sort by that score (FR-13).

    Scores are mapped back to candidates by **position in the provider's returned
    list**, matched to each returned doc's text, rather than by index position
    alone. The `RerankerProvider` protocol returns `ScoredDoc(chunk_id=...)` where
    the id is the provider's own positional label, not our `chunk_id` — trusting
    that label would attach scores to the wrong chunks.

    A provider that returns fewer results than submitted (its own `top_k`) leaves
    the remainder unscored rather than shifting scores onto wrong chunks. Those
    candidates then fall out of the reranked list, which is the conservative
    outcome: dropping a passage beats answering from an unranked one.
    """
    if not candidates:
        return []

    texts = [c.text for c in candidates]
    scored = reranker.rerank(query, texts, top_k=top_k)
    if not scored:
        log.warning("reranker returned no results; dropping all candidates")
        return []

    # Provider scores carry our chunk ids only when it was constructed with them.
    # The contract is positional, so text is the reliable join key.
    score_by_text: dict[str, float] = {}
    for hit in scored:
        score_by_text.setdefault(hit.text, hit.score)

    out: list[RetrievalCandidate] = []
    for candidate in candidates:
        score = score_by_text.get(candidate.text)
        if score is None:
            continue
        candidate.scores[Stage.RERANKED] = score
        out.append(candidate)

    out.sort(key=lambda c: (-c.scores[Stage.RERANKED], c.doc_id, c.chunk_index))
    for rank, candidate in enumerate(out, start=1):
        candidate.ranks[Stage.RERANKED] = rank

    log.debug(
        "reranked candidates",
        extra={
            "submitted": len(candidates),
            "returned": len(out),
            "top_k": top_k,
        },
    )
    return out