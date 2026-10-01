"""Tombstone semantics: disable, enable, supersede, delete (FR-7, FR-8).

`superseded`, `disabled`, and `deleted` are **tombstones, not deletes**. The row
and its chunks remain; retrieval filters on document state, so a tombstoned
document stops being served immediately. Physical removal is a separate
background operation, out of scope for v1.

The distinction that makes this cheap: the reason to tombstone rather than delete
is that "stop serving this now" and "remove the bytes" have different urgency.
The first must be instant and must not risk a partial index; the second can be
deferred and retried. Folding them together means either an instant operation is
slow, or a fast one leaves the system in a state where a document is half
removed and still being served.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models import Document, DocumentState
from app.ingest.keyword import refresh_keyword_state
from app.ingest.states import transition

log = get_logger("app.ingest.tombstone")


def get_document(session: Session, doc_id: str) -> Document:
    """Fetch a document or raise NotFoundError.

    Tombstoned documents are returned, not hidden. An admin needs to see that a
    document is disabled in order to re-enable it.
    """
    doc = session.get(Document, doc_id)
    if doc is None or doc.state == DocumentState.DELETED:
        raise NotFoundError(f"document {doc_id}")
    return doc


def disable(session: Session, doc_id: str) -> Document:
    """Stop serving a document. Chunks stay (FR-7)."""
    doc = get_document(session, doc_id)
    if doc.state == DocumentState.DISABLED:
        return doc
    doc = transition(session, doc, DocumentState.DISABLED)
    # The keyword rows are synced too. They hold a denormalised copy of the
    # state so the filter can apply inside the term scan, which means a state
    # change has to be pushed down or the copy serves a disabled document.
    refresh_keyword_state(session, doc_id)
    return doc


def enable(session: Session, doc_id: str) -> Document:
    """Resume serving a disabled document.

    Only `disabled` can be re-enabled. A `superseded` document has been replaced
    by a newer version; re-enabling it would serve two conflicting versions of
    the same policy, which is the failure FR-8 exists to prevent.
    """
    doc = get_document(session, doc_id)
    if doc.state == DocumentState.DISABLED:
        doc = transition(session, doc, DocumentState.LIVE)
        refresh_keyword_state(session, doc_id)
        return doc
    return doc


def supersede(
    session: Session, doc_id: str, *, replacement_doc_id: str | None = None
) -> Document:
    """Retire a document that a newer version replaces (FR-8).

    `replacement_doc_id` records which version took over, so an admin looking at
    a retired document can find its successor. Recorded rather than required: a
    document can be superseded by a manual edit that never produced a new row.
    """
    doc = get_document(session, doc_id)
    if doc.state != DocumentState.LIVE:
        return doc
    doc = transition(session, doc, DocumentState.SUPERSEDED)
    if replacement_doc_id is not None and replacement_doc_id != doc_id:
        doc.duplicate_of = replacement_doc_id
    refresh_keyword_state(session, doc_id)
    return doc


def mark_deleted(session: Session, doc_id: str) -> Document:
    """Tombstone a document as deleted.

    The row is kept so the deletion is auditable and so a re-upload of the same
    content is recognized (via the retained `content_hash`) rather than silently
    duplicating. `get_document` hides it from the API.
    """
    doc = session.get(Document, doc_id)
    if doc is None:
        raise NotFoundError(f"document {doc_id}")
    if doc.state == DocumentState.DELETED:
        return doc
    doc = transition(session, doc, DocumentState.DELETED)
    refresh_keyword_state(session, doc_id)
    return doc


def live_documents(session: Session) -> list[Document]:
    """All currently retrievable documents, newest first (FR-27)."""
    return list(
        session.execute(
            select(Document)
            .where(Document.state == DocumentState.LIVE)
            .order_by(Document.uploaded_at.desc())
        ).scalars()
    )


def all_documents(session: Session) -> list[Document]:
    """All documents including tombstones, for the admin view (FR-27)."""
    return list(
        session.execute(
            select(Document)
            .where(Document.state != DocumentState.DELETED)
            .order_by(Document.uploaded_at.desc())
        ).scalars()
    )
