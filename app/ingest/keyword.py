"""Keyword index for Phase 1.

Derived from the chunk store, never authoritative. Every row here can be dropped
and rebuilt from `chunks` (architecture.md 4.3), which is what makes index drift
a recoverable rebuild rather than a data-loss event.

The document state is denormalised onto each term row. A keyword query can then
restrict to live documents inside the index scan, so a tombstone takes effect
immediately instead of leaving stale matches servable until a reindex. `Chunk`
remains the only source of truth for what exists; this copy exists only to make
the filter cheap.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.db.models import Chunk, ChunkTerm, Document, DocumentState

# Words this short carry almost no retrieval signal ("the", "and", "of") but
# dominate term counts, so including them makes every document look similar to
# every other and pushes BM25 scores toward uniform.
_MIN_TERM_LENGTH = 2
_MAX_TERM_LENGTH = 64

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

#: Bounded vocabulary per chunk. A pathological input of a million distinct
#: tokens would otherwise produce a million rows for one passage; the top terms
#: by frequency are kept, which is what BM25 would weight anyway.
_MAX_TERMS_PER_CHUNK = 2000


def tokenize(text: str) -> list[str]:
    """Split text into lowercase keyword-index terms.

    Deliberately not a language-specific tokenizer: the corpus is user-supplied
    and its language is unknown, so anything requiring a language model would
    index some documents and silently under-index others. Word-boundary
    matching degrades gracefully on unknown languages.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if _MIN_TERM_LENGTH <= len(t) <= _MAX_TERM_LENGTH]


def write_keyword_index(session: Session, doc_id: str) -> int:
    """Rebuild the keyword index for one document. Returns rows written.

    The whole document is rebuilt rather than diffed. Chunk counts per document
    are small enough that a diff would cost more in complexity than it saves in
    writes, and a full rebuild cannot leave partial state behind — the failure
    mode of an incremental update.
    """
    session.execute(delete(ChunkTerm).where(ChunkTerm.chunk_id.in_(_chunk_ids(session, doc_id))))

    state = _document_state(session, doc_id)
    if state is None:
        # No document row: nothing to index. Reached when a delete removed the
        # document between chunking and indexing.
        return 0

    chunks = session.execute(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
    ).scalars().all()

    rows: list[ChunkTerm] = []
    for chunk in chunks:
        counts = Counter(tokenize(chunk.text))
        for term, count in counts.most_common(_MAX_TERMS_PER_CHUNK):
            rows.append(
                ChunkTerm(
                    chunk_id=chunk.chunk_id,
                    term=term[:_MAX_TERM_LENGTH],
                    term_count=count,
                    state=state,
                )
            )

    session.add_all(rows)
    session.flush()
    return len(rows)


def _chunk_ids(session: Session, doc_id: str) -> list[str]:
    return list(
        session.execute(select(Chunk.chunk_id).where(Chunk.doc_id == doc_id)).scalars()
    )


def _document_state(session: Session, doc_id: str) -> DocumentState | None:
    return session.execute(
        select(Document.state).where(Document.doc_id == doc_id)
    ).scalar_one_or_none()


def purge_keyword_index(session: Session, doc_id: str) -> int:
    """Remove all keyword rows for a document. Used on re-ingest and delete."""
    chunk_ids = _chunk_ids(session, doc_id)
    if not chunk_ids:
        return 0
    result = session.execute(delete(ChunkTerm).where(ChunkTerm.chunk_id.in_(chunk_ids)))
    # A DML DELETE returns a CursorResult; the declared `Result` type is wider,
    # and `rowcount` is -1 on dialects that cannot report it.
    return max(getattr(result, "rowcount", 0) or 0, 0)


def refresh_keyword_state(session: Session, doc_id: str) -> None:
    """Copy the current document state onto its keyword rows.

    Called after a state transition. Cheap relative to a rebuild, and it is what
    makes disable/delete effective on the keyword path immediately (FR-7).
    """
    state = _document_state(session, doc_id)
    if state is None:
        return
    chunk_ids = _chunk_ids(session, doc_id)
    if not chunk_ids:
        return
    session.execute(
        update(ChunkTerm).where(ChunkTerm.chunk_id.in_(chunk_ids)).values(state=state)
    )


class ChunkTermRow:
    """Result of a keyword lookup: chunk identity plus a raw term-frequency sum.

    Scoring proper is Phase 2 (BM25 over corpus statistics). This returns the
    deterministic components it needs so ranking lives in one place.
    """

    __slots__ = ("chunk_id", "chunk_index", "doc_id", "matched_terms", "score")

    def __init__(
        self,
        chunk_id: str,
        doc_id: str,
        chunk_index: int,
        score: float,
        matched_terms: Iterable[str],
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.score = score
        self.matched_terms = list(matched_terms)


def keyword_lookup(
    session: Session, terms: Sequence[str], *, limit: int = 50
) -> list[ChunkTermRow]:
    """Find live chunks containing any of `terms`, best raw match first.

    The `state == LIVE` predicate is applied here and not left to the caller.
    Phase 2 retrieval will also apply `retrievable_chunk_filter()`, but a
    tombstoned document must never be servable from this path even if that is
    missed, so the filter is repeated deliberately.
    """
    if not terms:
        return []

    normalized = [t.lower()[:_MAX_TERM_LENGTH] for t in terms]
    rows = session.execute(
        select(
            ChunkTerm.chunk_id,
            ChunkTerm.term,
            ChunkTerm.term_count,
            Chunk.doc_id,
            Chunk.chunk_index,
        )
        .join(Chunk, Chunk.chunk_id == ChunkTerm.chunk_id)
        .where(
            ChunkTerm.term.in_(normalized),
            ChunkTerm.state == DocumentState.LIVE,
        )
    ).all()

    grouped: dict[str, dict] = {}
    for chunk_id, term, term_count, doc_id, chunk_index in rows:
        entry = grouped.setdefault(
            chunk_id,
            {"doc_id": doc_id, "chunk_index": chunk_index, "score": 0, "terms": []},
        )
        # Repeated query terms are not double counted: the same term matching a
        # chunk twice should not outrank a chunk matching two distinct terms,
        # which is the signal BM25 is meant to reward.
        if term not in entry["terms"]:
            entry["terms"].append(term)
            entry["score"] += float(term_count)

    results = [
        ChunkTermRow(
            chunk_id=chunk_id,
            doc_id=entry["doc_id"],
            chunk_index=entry["chunk_index"],
            score=entry["score"],
            matched_terms=entry["terms"],
        )
        for chunk_id, entry in grouped.items()
    ]
    # Sorted by score, then by (doc_id, chunk_index) so equal scores return a
    # stable order. Without the tiebreak, result order depends on scan order and
    # tests and citation rendering become non-deterministic.
    results.sort(key=lambda r: (-r.score, r.doc_id, r.chunk_index))
    return results[:limit]