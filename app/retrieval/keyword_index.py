"""Keyword index: the BM25 half of hybrid retrieval (FR-12).

architecture.md §3.3 singles out this stage as the reason hybrid search is not
optional: vector similarity is bad at exact identifiers, and document Q&A lives
on error codes, part numbers, surnames, dates, and "Article 12". Dropping the
keyword stage produces confidently wrong answers precisely on the queries users
trust most.

Implementation note — **BM25 runs in Python over the `chunk_terms` table, not in
the database.** For the target scale (NFR-4, 1M chunks) Postgres full-text search
with a database-computed ranking is the right call, and the `KeywordIndex`
protocol below is the seam where that implementation slots in without touching
callers. The variant here is correct and testable but recomputes corpus
statistics per query; `docs/eval_results.md` records the measured cost. That gap
is why this file must not be mistaken for the production keyword index.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, ChunkTerm, Document, DocumentState
from app.ingest.keyword import tokenize
from app.retrieval.types import AccessFilter, RetrievalCandidate, Stage
from app.retrieval.vector_store import acl_sql_predicate

log = logging.getLogger(__name__)

#: Okapi BM25 parameters. k1 controls term-frequency saturation and b controls
#: length normalisation. k1=1.2, b=0.75 are the values from the original BM25
#: paper and are used unchanged so results are comparable with the literature
#: rather than tuned against our own corpus and therefore uninterpretable.
_K1 = 1.2
_B = 0.75

#: Scanned rows per query term. Distinct chunks are collected until this many
#: have been seen, which bounds work per query without letting a single very
#: common term scan the whole index.
_SCAN_ROWS_PER_TERM = 4000

#: Longest n-gram length, matching the identifier forms this stage exists to
#: catch (`Article 12`, `POL-REF-4821`).
_MAX_NGRAM = 3

#: Query-side stopwords. Two distinct jobs, both easy to get wrong:
#:
#: 1. They carry no retrieval signal, so they only dilute the discriminative
#:    terms.
#: 2. They must be dropped *before* n-grams are formed. Stopword bigrams like
#:    `of_the` are rare terms in any real corpus, so BM25 gives them a high IDF
#:    and a query consisting only of stopwords ends up scoring strongly on
#:    `the_of` while meaning nothing. That is not hypothetical: it is what this
#:    list exists to prevent, asserted by
#:    `test_ubiquitous_terms_score_below_a_discriminative_term`.
_QUERY_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at",
        "for", "with", "from", "by", "as", "is", "are", "was", "were", "be",
        "been", "being", "do", "does", "did", "can", "could", "would",
        "should", "will", "shall", "may", "might", "must", "it", "its", "this",
        "that", "these", "those", "there", "here", "what", "which", "who",
        "whom", "whose", "when", "where", "why", "how", "if", "then", "than",
        "so", "such", "not", "no", "my", "your", "our", "their", "me", "you",
    }
)


def _ngrams(tokens: Sequence[str]) -> list[str]:
    """Unigrams plus adjacent bigrams and trigrams, over content terms only.

    Phrases matter more than they look: a chunk containing both "refund" and
    "window" should outrank one containing "refund" alone when the query asks
    about the refund *window*. Unigrams alone cannot express that adjacency, and
    it is cheap to add here rather than in the query language.

    Stopwords are removed before n-grams are formed, not after. Forming n-grams
    over them first produces rare terms (`of_the`) whose high IDF makes a
    contentless query score as though it were a precise one.
    """
    content = [t for t in tokens if t not in _QUERY_STOPWORDS]
    if not content:
        return []
    out = list(content)
    for size in range(2, _MAX_NGRAM + 1):
        for i in range(len(content) - size + 1):
            out.append("_".join(content[i : i + size]))
    return out


@runtime_checkable
class KeywordIndex(Protocol):
    """Protocol for the lexical half of hybrid retrieval (architecture.md §7.3)."""

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        """Return up to `k` chunks ranked by BM25, access filter pushed down."""
        ...


class Bm25KeywordIndex:
    """Okapi BM25 over the derived `chunk_terms` table.

    Ranking is computed in Python from the frequency rows the ingestion pipeline
    already wrote, so there is no second index to keep in sync: `chunk_terms` is
    rebuilt from `chunks` on every ingest, and this class only reads it.

    The cost model is the honest part. IDF needs corpus-wide document frequency
    and average document length, so each query issues a small aggregate over
    `chunk_terms` before scoring. That is fine against the local corpus and is the
    reason the production variant must push BM25 into the database.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- corpus statistics -------------------------------------------------

    def _corpus_stats(self, terms: Sequence[str]) -> tuple[dict[str, float], float, int]:
        """Per-term IDF, average chunk length, and total live chunk count.

        Only live documents are counted. Counting tombstoned chunks in the
        statistics would let a disabled document depress the IDF of a term that
        still appears in live content, quietly changing ranking scores for every
        query after a disable — the kind of change that is invisible until someone
        asks why relevance shifted.
        """
        total_chunks = (
            self._session.execute(
                select(func.count(func.distinct(ChunkTerm.chunk_id)))
                .join(Chunk, Chunk.chunk_id == ChunkTerm.chunk_id)
                .join(Document, Document.doc_id == Chunk.doc_id)
                .where(Document.state == DocumentState.LIVE)
            )
            .scalar()
            or 0
        )
        if total_chunks == 0:
            return {}, 0.0, 0

        df_rows = self._session.execute(
            select(ChunkTerm.term, func.count(func.distinct(ChunkTerm.chunk_id)))
            .join(Chunk, Chunk.chunk_id == ChunkTerm.chunk_id)
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(
                ChunkTerm.term.in_(list(terms)),
                Document.state == DocumentState.LIVE,
            )
            .group_by(ChunkTerm.term)
        ).all()

        idf: dict[str, float] = {}
        for term, doc_freq in df_rows:
            # Smoothed IDF, floored at 0: a term in every chunk carries no
            # discriminative signal, and the unsmoothed form goes slightly
            # negative, which would reward matching it.
            df = float(doc_freq)
            idf[term] = max(
                math.log(1.0 + (total_chunks - df + 0.5) / (df + 0.5)), 0.0
            )

        # BM25's length normalisation is defined against average chunk length,
        # not the average count of matched terms, so query the true chunk length
        # average over live documents.
        avg_len = float(
            self._session.execute(
                select(func.avg(Chunk.token_count))
                .join(Document, Document.doc_id == Chunk.doc_id)
                .where(Document.state == DocumentState.LIVE)
            ).scalar()
            or 0.0
        )
        return idf, avg_len, total_chunks

    # -- search ------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        tokens = tokenize(query)
        if not tokens:
            return []

        terms = _ngrams(tokens)
        idf, avg_len, total_chunks = self._corpus_stats(terms)
        if total_chunks == 0 or not idf:
            return []
        if avg_len <= 0.0:
            # Every live chunk is empty of tokens. Any normalisation against a
            # zero average would divide by zero; treat all lengths as equal.
            avg_len = 1.0

        query_clause = ChunkTerm.term.in_(list(idf.keys()))
        filters = [ChunkTerm.state == DocumentState.LIVE, query_clause]

        stmt = (
            select(
                ChunkTerm.chunk_id,
                ChunkTerm.term,
                ChunkTerm.term_count,
                Chunk.doc_id,
                Chunk.chunk_index,
                Chunk.text,
                Chunk.breadcrumb,
                Chunk.page,
                Chunk.char_start,
                Chunk.char_end,
                Chunk.token_count,
                Document.acl_tags,
            )
            .join(Chunk, Chunk.chunk_id == ChunkTerm.chunk_id)
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(*filters)
        )
        acl = acl_sql_predicate(self._session, access, Document.acl_tags)
        if acl is not None:
            stmt = stmt.where(acl)

        rows = self._session.execute(stmt.limit(len(terms) * _SCAN_ROWS_PER_TERM)).all()

        # term -> {chunk_id: frequency}, then score each chunk once.
        postings: dict[str, dict[str, float]] = defaultdict(dict)
        meta: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            chunk_id = row[0]
            postings[row[1]][chunk_id] = float(row[2])
            meta[chunk_id] = row[3:]

        scores: dict[str, float] = defaultdict(float)
        for term, term_idf in idf.items():
            postings_for_term = postings.get(term)
            if not postings_for_term:
                continue
            for chunk_id, freq in postings_for_term.items():
                chunk_len = float(meta[chunk_id][7] or 0) or 1.0
                # Okapi saturation: repeated terms stop mattering quickly, so a
                # chunk that repeats the word ten times does not outrank one that
                # matches a second distinct query term.
                numerator = freq * (_K1 + 1.0)
                denominator = freq + _K1 * (
                    1.0 - _B + _B * (chunk_len / avg_len)
                )
                scores[chunk_id] += term_idf * (numerator / denominator)

        if not scores:
            return []

        ranked = sorted(
            scores.items(),
            key=lambda pair: (-pair[1], meta[pair[0]][0], meta[pair[0]][1]),
        )

        out: list[RetrievalCandidate] = []
        for chunk_id, score in ranked:
            doc_id, chunk_index, text, breadcrumb, page, start, end, tok, acl = meta[chunk_id]
            if access is not None and not access.permits(list(acl or [])):
                # Defence in depth: the SQL predicate above is a LIKE on SQLite
                # and can match a tag as a substring. This is the authoritative
                # check and runs before the limit.
                continue
            candidate = RetrievalCandidate(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=text,
                breadcrumb=breadcrumb,
                page=page,
                char_start=start,
                char_end=end,
                token_count=tok,
                acl_tags=list(acl or []),
            )
            candidate.scores[Stage.KEYWORD] = score
            out.append(candidate)
            if len(out) >= k:
                break

        log.debug(
            "keyword search",
            extra={
                "terms": len(terms),
                "access": access.describe() if access else "unrestricted",
                "returned": len(out),
            },
        )
        return out

    def matched_terms(self, query: str) -> list[str]:
        """The n-gram terms this index would search for. Used by tests and logs."""
        return _ngrams(tokenize(query))


def build_keyword_index(session: Session) -> KeywordIndex:
    """Return the configured keyword index.

    The dialect is not branched on here: the frequency rows are portable, and the
    production variant that pushes BM25 into the database would be selected by the
    same kind of check `build_vector_store` uses.
    """
    return Bm25KeywordIndex(session)