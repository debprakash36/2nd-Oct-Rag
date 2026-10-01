"""Content hashing, dedupe, and version supersede (FR-4, FR-8).

The hash is computed over the **chunked content**, not the file bytes
(architecture.md 4.4). Re-saving a PDF changes the bytes without changing the
text; byte hashing would re-embed an identical document for nothing, and would
miss the case that actually matters — the same text arriving in a different file.

This is also what makes the pipeline idempotent under queue redelivery. The
queue delivers at least once, so a redelivered message must converge rather than
accumulate. Keying on content means a repeat is recognized as a repeat.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.db.models import Document, DocumentState

log = get_logger("app.ingest.dedupe")


class HashVerdict(Enum):
    """What a content hash turned out to mean."""

    NEW = "new"
    """No document has this content. Ingest it."""

    SAME_DOCUMENT = "same_document"
    """This exact content is already indexed on this document. No-op (FR-8, idempotency)."""

    NEWER_VERSION = "newer_version"
    """Same document, revised content. Supersede the old version (FR-8)."""

    DUPLICATE = "duplicate"
    """Same content, different document. Flag it; let an admin decide (FR-4)."""


@dataclass(frozen=True)
class HashCheck:
    """Result of comparing a content hash against the corpus."""

    verdict: HashVerdict
    existing_doc_id: str | None
    existing_version: int | None


def compute_content_hash(chunk_texts: list[str]) -> str:
    """Hash the chunked content of a document.

    Ordered and length-prefixed so that `["ab", "c"]` and `["a", "bc"]` do not
    collide. A naive join would hash both to the same value, and two different
    documents would be treated as one.
    """
    hasher = hashlib.sha256()
    for text in chunk_texts:
        encoded = text.encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def check_content_hash(
    session: Session, content_hash: str, doc_id: str
) -> HashCheck:
    """Classify a content hash against existing documents.

    Ordering matters here, twice over.

    More than one document can share a content hash — that is precisely the
    duplicate case — so the query cannot assume a single row. Rows are ordered to
    put the most informative one first: the document itself, then a live copy,
    then a tombstone. `scalar_one_or_none` would raise on exactly the case this
    function exists to detect.

    A tombstoned document must not be treated as an existing live copy, or
    re-uploading a file whose old version was disabled would be rejected, and a
    deleted document would permanently block its own re-upload.
    """
    rows = session.execute(
        select(Document)
        .where(Document.content_hash == content_hash)
        .order_by(
            # Priority 0: the document being ingested (self-match).
            (Document.doc_id == doc_id).desc(),
            # Priority 1: a live document, which is a real duplicate.
            (Document.state == DocumentState.LIVE).desc(),
            Document.uploaded_at.desc(),
        )
    ).scalars().all()

    for row in rows:
        if row.doc_id == doc_id:
            # Same document, same content. This is the redelivery case.
            if row.state == DocumentState.LIVE:
                return HashCheck(HashVerdict.SAME_DOCUMENT, row.doc_id, row.version)
            # Tombstoned or failed: same document re-uploaded. Treat as a
            # re-ingestion rather than a duplicate of itself.
            return HashCheck(HashVerdict.NEWER_VERSION, row.doc_id, row.version)

    live = next((r for r in rows if r.state == DocumentState.LIVE), None)
    if live is not None:
        return HashCheck(HashVerdict.DUPLICATE, live.doc_id, live.version)

    if not rows:
        return HashCheck(HashVerdict.NEW, None, None)

    # Only tombstoned copies exist, so this is not a live duplicate.
    return HashCheck(HashVerdict.NEW, None, None)


def is_near_duplicate(session: Session, doc_id: str) -> bool:
    """Whether a document is flagged as a duplicate of another (FR-4).

    Recorded on the row rather than rejected, so an admin can decide. Silently
    dropping an upload looks like data loss from the user's side, and silently
    ingesting it produces two documents that answer questions identically.
    """
    doc = session.get(Document, doc_id)
    return doc is not None and doc.duplicate_of is not None
