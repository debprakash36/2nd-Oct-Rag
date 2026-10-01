"""Reranker provider implementations (NFR-10 seam for FR-13).

`get_reranker_provider()` is the single place a concrete reranker is chosen,
mirroring `app/providers/embedding.py::get_embedding_provider`.

**Neither provider here is a trained cross-encoder.** `LexicalRerankerProvider` is
a deterministic term-overlap scorer; `NullRerankerProvider` passes scores through
as zero. A real cross-encoder is a Phase 2 follow-up, blocked here on model
access rather than on design — `docs/eval_results.md` states the consequence for
what the measured numbers prove.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence

from app.core.config import Settings, get_settings
from app.ingest.keyword import tokenize
from app.providers.base import RerankerProvider, ScoredDoc

log = logging.getLogger(__name__)

#: Terms that appear in most chunks and discriminate nothing. Counted in the
#: length normalisation but given no overlap credit.
_RERANK_STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is",
        "are", "was", "were", "be", "been", "this", "that", "these", "those",
        "it", "its", "as", "at", "by", "with", "from", "not", "but",
    }
)


class LexicalRerankerProvider:
    """Deterministic term-overlap reranker. Offline stand-in for a cross-encoder.

    Scores a document by the length-normalised coverage of the query's
    discriminative terms, weighted toward rare terms. The shape is a lexical
    similarity rather than a learned one, so it rewards surface overlap that the
    BM25 stage has already exploited.

    Its actual value in the pipeline is different, and it is why this is not just
    wasted work: the vector half of the corpus here uses a hash-based embedding
    provider (`app/providers/embedding.py::FakeEmbeddingProvider`) that carries no
    real semantics. Reranking on lexical overlap recovers some of the signal that
    the fake embeddings cannot provide, which is what lets the retrieval numbers
    mean anything at all in this environment.

    Two corrections over a plain coverage count, both of which were visible as
    measured retrieval failures rather than theorised:

    * **Coverage is weighted by term rarity.** An unweighted mean over query
      terms is dominated by the common ones, so a query sharing "policy" and
      "stated" with every document in the corpus scored the same against all of
      them. Weighting by inverse document frequency over the candidate set is
      what lets the one distinguishing term decide the order.
    * **A term present in most candidates is evidence of nothing.** Distinctive
      is only meaningful relative to the pool; if 90% of candidates contain a
      term, its presence cannot separate them and it is dropped from the query's
      discriminating set.

    The score is in [0, 1] so it is comparable with the cosine similarity the
    threshold gate also sees, and so `RETRIEVAL_THRESHOLD` stays interpretable
    regardless of which reranker is configured.
    """

    #: A term appearing in more than this share of candidates cannot discriminate
    #: between them, so it is excluded from the coverage computation.
    _COMMON_TERM_RATIO = 0.90

    def rerank(self, query: str, docs: Sequence[str], *, top_k: int) -> list[ScoredDoc]:
        query_terms = [t for t in tokenize(query) if t not in _RERANK_STOPWORDS]
        if not query_terms:
            # An all-stopword query ("what about that?") carries no signal. Score
            # everything zero and preserve input order rather than inventing a
            # preference; the fusion stage already established the ranking.
            return [
                ScoredDoc(chunk_id=str(i), score=0.0, text=doc)
                for i, doc in enumerate(docs[:top_k])
            ]

        tokenized = [tokenize(doc) for doc in docs]
        weights, query_set = self._weighted_query(query_terms, tokenized)

        results: list[ScoredDoc] = []
        for i, (doc, doc_terms) in enumerate(zip(docs, tokenized, strict=True)):
            if not doc_terms:
                results.append(ScoredDoc(chunk_id=str(i), score=0.0, text=doc))
                continue

            counts = Counter(doc_terms)
            matched = 0.0
            for term in query_set:
                if counts.get(term):
                    matched += weights[term]

            coverage = matched / sum(weights.values()) if weights else 0.0

            # Bounded length preference: prefer the shorter of two equally
            # covering passages, but only mildly (log, not linear) so a genuinely
            # relevant long chunk is not buried by a short keyword-stuffed one.
            length_penalty = 1.0 / (1.0 + math.log1p(len(doc_terms) / 200.0))

            # Coverage dominates: it is the signal, and it is bounded. The
            # adjacency bonus rewards the query's terms appearing together,
            # which a bag-of-words count cannot express.
            adjacency = self._adjacency_bonus(doc_terms, query_set)
            score = min(1.0, coverage * 0.8 + adjacency * 0.2) * (0.6 + 0.4 * length_penalty)
            results.append(ScoredDoc(chunk_id=str(i), score=score, text=doc))

        results.sort(key=lambda s: (-s.score, s.chunk_id))
        return results[:top_k]

    def _weighted_query(
        self, query_terms: Sequence[str], tokenized: Sequence[Sequence[str]]
    ) -> tuple[dict[str, float], set[str]]:
        """Inverse-frequency weights over the candidate pool.

        Weights are computed against the candidates being ranked rather than a
        corpus-wide index because that is the only pool available at this stage
        and it is the pool that matters: a term shared by every candidate cannot
        reorder them, so including it only dilutes the terms that can.
        """
        pool = len(tokenized)
        if pool == 0:
            return {t: 1.0 for t in query_terms}, set(query_terms)

        doc_freq: Counter[str] = Counter()
        for doc_terms in tokenized:
            doc_freq.update(set(doc_terms))

        weights: dict[str, float] = {}
        for term in set(query_terms):
            df = doc_freq.get(term, 0)
            if df / pool > self._COMMON_TERM_RATIO:
                continue
            # +1 smoothing keeps a term present in every candidate from
            # collapsing to zero weight and silently dropping a real signal.
            weights[term] = math.log(1.0 + pool / (1.0 + df))

        if not weights:
            # Every query term is in every candidate. There is no basis to
            # reorder, so score uniformly and let fusion's ordering stand.
            return {t: 1.0 for t in query_terms}, set(query_terms)
        return weights, set(weights)

    def _adjacency_bonus(
        self, doc_terms: Sequence[str], query_set: set[str], *, window: int = 5
    ) -> float:
        """Fraction of query terms found within `window` of another query term.

        Rewards "refund window" over a document that merely contains "refund" and
        "window" on opposite pages.
        """
        positions = [i for i, t in enumerate(doc_terms) if t in query_set]
        if len(positions) < 2:
            return 0.0
        close = 0
        for i, pos in enumerate(positions):
            for other in positions[i + 1 :]:
                if other - pos <= window:
                    close += 1
        return min(1.0, close / max(len(positions) - 1, 1))


class NullRerankerProvider:
    """Pass-through reranker that scores everything 0.0.

    Retained for two real uses: measuring the pipeline with the rerank stage
    disabled-but-present (§10 check 2), and running a deployment that has enabled
    reranking before a real provider exists. In the latter case every score is
    below any positive relevance threshold, so the pipeline abstains — no answer
    rather than an unranked one, which is the direction §7.3 argues for.

    Note this is *not* the same as `RETRIEVAL_RERANK_ENABLED=false`. That skips
    the stage entirely and preserves the fused order; this preserves the order
    only through the (doc_id, chunk_index) tiebreak.
    """

    def rerank(self, query: str, docs: Sequence[str], *, top_k: int) -> list[ScoredDoc]:
        return [
            ScoredDoc(chunk_id=str(i), score=0.0, text=doc)
            for i, doc in enumerate(docs[:top_k])
        ]


def get_reranker_provider(settings: Settings | None = None) -> RerankerProvider:
    """Return the configured reranker (NFR-10 seam).

    Phase 2 replaces the lexical stand-in with a real cross-encoder; the ablation
    harness (`eval/run_eval.py --ablate rerank`) depends on this being swappable.
    """
    s = settings or get_settings()
    if getattr(s, "reranker_provider", "lexical") == "null":
        return NullRerankerProvider()
    return LexicalRerankerProvider()