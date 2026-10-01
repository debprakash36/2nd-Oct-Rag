"""Stage 3 post-fusion: relevance threshold, dedupe ordering, context budget.

Two separate mechanisms live here and they are easy to confuse:

* The **threshold** (FR-14) is a quality gate on the *top* score. Below it, the
  pipeline abstains with no LLM call at all. architecture.md §3.3 calls this a
  measured parameter rather than a guess, and §10 check 4 requires sweeping it
  against the eval set to hit recall@10 ≥ 0.85 and a 10-30% refusal rate
  *simultaneously*.
* The **token budget** (FR-15) is a capacity constraint, not a quality gate. It
  evicts lowest-scoring chunks until the context fits. Evicting a chunk is not a
  statement about its relevance, and a run that fits in budget while still
  abstaining is a different outcome from one that evicts and answers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.retrieval.types import RetrievalCandidate, Stage

log = logging.getLogger(__name__)

#: Above this many tokens a single passage is treated as unciteable. A passage
#: that fills the entire context window leaves nowhere for other evidence and
#: nothing for the generator to say about it. These passages are dropped with a
#: log line rather than passed to the model to summarise.
MAX_CHUNK_TOKENS = 2000


def apply_threshold(
    candidates: Sequence[RetrievalCandidate], threshold: float
) -> tuple[list[RetrievalCandidate], bool, str]:
    """Gate on the top candidate's best score (FR-14).

    Returns `(kept, abstained, reason)`.

    Only the top score is tested, not every candidate: the threshold answers
    "is the best thing we found relevant enough to answer from", and filtering
    every candidate by the same bar would discard a strong supporting passage
    whenever the top one barely passed.
    """
    if not candidates:
        return [], True, "no candidates returned by any stage"

    top = candidates[0].best_score()
    if top < threshold:
        log.info(
            "below relevance threshold",
            extra={"top_score": round(top, 6), "threshold": threshold},
        )
        return [], True, f"top score {top:.4f} below threshold {threshold:.4f}"

    kept = list(candidates)
    for candidate in kept:
        if candidate.token_count > MAX_CHUNK_TOKENS:
            log.info(
                "dropping oversized chunk",
                extra={"chunk_id": candidate.chunk_id, "tokens": candidate.token_count},
            )
    kept = [c for c in kept if c.token_count <= MAX_CHUNK_TOKENS]

    # The gate was passed by the top candidate; if that candidate was the only one
    # and it was oversized, there is nothing left to answer from.
    if not kept:
        return [], True, "only candidate exceeded the per-chunk token cap"
    return kept, False, ""


def apply_token_budget(
    candidates: Sequence[RetrievalCandidate],
    max_context_tokens: int,
    *,
    reserve_for_answer: int = 0,
) -> tuple[list[RetrievalCandidate], int]:
    """Evict lowest-scoring candidates until the context fits (FR-15).

    `reserve_for_answer` holds back tokens for the generated answer. A budget that
    counts only retrieved context lets the model run out of room mid-answer,
    producing a truncated response that looks like a retrieval success.

    Eviction is from the *bottom* of the relevance-sorted list, so the highest
    ranked passages always survive. Returns `(kept, evicted_count)`.
    """
    available = max_context_tokens - reserve_for_answer
    if available <= 0:
        log.warning(
            "context budget exhausted by answer reserve",
            extra={"max_context_tokens": max_context_tokens, "reserve": reserve_for_answer},
        )
        return [], len(candidates)

    kept: list[RetrievalCandidate] = []
    used = 0
    for candidate in candidates:
        cost = candidate.token_count or estimate_tokens(candidate.text)
        if used + cost > available:
            continue
        kept.append(candidate)
        used += cost

    evicted = len(candidates) - len(kept)
    if evicted:
        log.debug(
            "evicted candidates to fit context budget",
            extra={"evicted": evicted, "used_tokens": used, "available": available},
        )
    return kept, evicted


def estimate_tokens(text: str) -> int:
    """Approximate token count when the index did not record one.

    Roughly four characters per token. Used only as a fallback for rows written
    before `token_count` was populated; the stored value is authoritative
    everywhere else because a real tokenizer count is what the generator's own
    budget was measured against.
    """
    return max(1, len(text) // 4)


def score_for_threshold(candidates: Sequence[RetrievalCandidate]) -> float:
    """The score the threshold gate will test.

    Reranked when available, else fused, else the best single-stage score. Kept in
    one place so the eval harness and the pipeline cannot disagree about which
    number was gated — a disagreement there would make a threshold sweep look
    like it achieved two things at once.
    """
    if not candidates:
        return 0.0
    return candidates[0].best_score()


def stage_of_score(candidate: RetrievalCandidate) -> Stage | None:
    """Which stage produced the score the threshold used. For eval reporting."""
    for stage in (Stage.RERANKED, Stage.FUSED, Stage.VECTOR, Stage.KEYWORD):
        if stage in candidate.scores:
            return stage
    return None