"""Stage 2 of retrieval: Reciprocal Rank Fusion and near-duplicate dedupe.

architecture.md §3.3 gives the reasoning for RRF and it is worth restating
because it constrains everything in this module: vector cosine similarity and
BM25 are not on comparable scales, so any weighted score-combination has to
calibrate the two scales against each other. That calibration is silently
invalidated whenever the corpus changes. RRF consumes only ranks, so a reindex
cannot make previously-tuned weights wrong.

The dedupe here is a retrieval-stage concern, not a cleanup pass: boilerplate
repeated across documents (headers, disclaimers, nav fragments) will otherwise
occupy the entire narrow context window with a single fact repeated K times
(architecture.md §3.3, FR-15).
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Sequence

from app.retrieval.types import RetrievalCandidate, Stage

log = logging.getLogger(__name__)

#: RRF damping constant (k=60 in Cormack et al. 2009). Large relative to the
#: candidate counts this system fetches, which is the intent: it flattens the
#: contribution difference between rank 1 and rank 5 so a single strong list
#: cannot dominate the fusion.
RRF_K = 60

#: Fingerprint window in characters. Long enough that two genuinely distinct
#: passages rarely collide, short enough that a shared boilerplate block matches.
_DEDUPE_WINDOW = 400

_NORMALISE_RE = re.compile(r"[^a-z0-9]+")


def _fingerprint(text: str) -> str:
    """Hashable signature of a chunk's leading normalised text.

    Casefolded with non-alphanumerics collapsed to a single space. Whitespace and
    casing are the differences that separate a duplicated boilerplate block from a
    genuinely distinct passage, so normalising them is what makes this a
    near-duplicate check rather than an exact-match check.
    """
    return _NORMALISE_RE.sub(" ", text[:_DEDUPE_WINDOW].lower()).strip()


def reciprocal_rank_fusion(
    ranked_lists: dict[Stage, Sequence[RetrievalCandidate]],
    *,
    rrf_k: int = RRF_K,
    weights: dict[Stage, float] | None = None,
) -> list[RetrievalCandidate]:
    """Fuse ranked candidate lists by Reciprocal Rank Fusion.

    Score for a candidate is `sum(weight / (rrf_k + rank))` over every list that
    returned it. Candidates appearing in only one list are not penalised against
    a phantom zero — they simply accumulate from that one list, which is why a
    chunk found by keyword alone can still outrank a chunk ranked middlingly by
    both.

    Each input list is expected in descending rank order; position in the list
    *is* the rank. The returned list is sorted by fused score descending with a
    stable (doc_id, chunk_index) tiebreak, because an unstable order makes
    citations and eval scores vary run to run.
    """
    weights = weights or {}
    fused: dict[str, RetrievalCandidate] = {}
    totals: dict[str, float] = defaultdict(float)

    for stage, candidates in ranked_lists.items():
        if not candidates:
            continue
        weight = weights.get(stage, 1.0)
        for position, candidate in enumerate(candidates, start=1):
            existing = fused.get(candidate.chunk_id)
            if existing is None:
                # Keep the first-seen copy's metadata; all copies are hydrated
                # from the same authoritative row, so they agree.
                fused[candidate.chunk_id] = candidate
                existing = candidate
            existing.ranks[stage] = position
            totals[candidate.chunk_id] += weight / (rrf_k + position)

    out: list[RetrievalCandidate] = []
    for chunk_id, candidate in fused.items():
        candidate.scores[Stage.FUSED] = totals[chunk_id]
        out.append(candidate)

    out.sort(key=lambda c: (-c.scores[Stage.FUSED], c.doc_id, c.chunk_index))
    return out


def deduplicate(
    candidates: Sequence[RetrievalCandidate],
    *,
    limit: int | None = None,
) -> list[RetrievalCandidate]:
    """Drop near-duplicate chunks, keeping the first occurrence in rank order.

    Input order is assumed to be descending relevance. Keeping the first means
    the surviving copy is the highest-ranked one, so dedupe cannot demote a
    passage the retriever was most confident about.

    An empty or near-empty fingerprint is not treated as a duplicate. Short chunks
    ("See section 4.") collapse to similar normalised strings by coincidence, and
    dropping those would silently lose real content.
    """
    seen: set[str] = set()
    out: list[RetrievalCandidate] = []
    for candidate in candidates:
        fingerprint = _fingerprint(candidate.text)
        if fingerprint and fingerprint in seen:
            continue
        if fingerprint:
            seen.add(fingerprint)
        out.append(candidate)
        if limit is not None and len(out) >= limit:
            break

    dropped = len(candidates) - len(out)
    if dropped:
        log.debug("deduped near-duplicate chunks", extra={"dropped": dropped})
    return out