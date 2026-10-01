"""Document state machine (FR-5, architecture.md 4.1).

An explicit transition table rather than scattered `if` statements. Two reasons:

1. Testability. `test_state_machine.py` asserts the table is exhaustive and
   legal, which is only possible if the rules live in one data structure.
2. A silent no-op is the dangerous failure here. If an illegal transition were
   ignored, a document would sit in a transient state with no way to tell
   whether the work happened. `transition()` raises instead.

Invariants the table encodes
----------------------------
* Only `LIVE` is retrievable.
* Any pipeline state can fail, with a reason.
* `superseded` and `disabled` are tombstones, not deletes. The row and its chunks
  remain; retrieval filters on state, so the document stops being served
  immediately. Physical removal is a separate background operation.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.errors import StateTransitionError
from app.core.logging import get_logger
from app.db.models import Document, DocumentState

log = get_logger("app.ingest.states")

#: The only legal transitions. Anything absent from this table is rejected.
TRANSITIONS: dict[DocumentState, frozenset[DocumentState]] = {
    DocumentState.PENDING: frozenset(
        {DocumentState.EXTRACTING, DocumentState.FAILED}
    ),
    DocumentState.EXTRACTING: frozenset(
        {DocumentState.CHUNKING, DocumentState.DUPLICATE, DocumentState.FAILED}
    ),
    DocumentState.CHUNKING: frozenset(
        {DocumentState.EMBEDDING, DocumentState.DUPLICATE, DocumentState.FAILED}
    ),
    DocumentState.EMBEDDING: frozenset(
        {DocumentState.INDEXING, DocumentState.FAILED}
    ),
    DocumentState.INDEXING: frozenset(
        {DocumentState.LIVE, DocumentState.FAILED}
    ),
    # Terminal in the same sense as `SUPERSEDED`: an admin can promote it back to
    # `PENDING` to index it deliberately, which is the "let an admin decide" half of
    # FR-4. The pipeline never indexes a duplicate on its own.
    DocumentState.DUPLICATE: frozenset(
        {DocumentState.PENDING, DocumentState.DELETED}
    ),
    DocumentState.LIVE: frozenset(
        {DocumentState.SUPERSEDED, DocumentState.DISABLED, DocumentState.DELETED,
         DocumentState.FAILED}
    ),
    # A re-ingest of a failed or tombstoned document restarts the pipeline.
    DocumentState.FAILED: frozenset({DocumentState.PENDING, DocumentState.DELETED}),
    DocumentState.SUPERSEDED: frozenset({DocumentState.DELETED}),
    DocumentState.DISABLED: frozenset({DocumentState.LIVE, DocumentState.DELETED}),
    DocumentState.DELETED: frozenset(),
}

#: States in which a document is retrievable. Enforced here so the rule has one
#: definition rather than being restated at each retrieval call site.
RETRIEVABLE_STATES = frozenset({DocumentState.LIVE})


def can_transition(source: DocumentState, target: DocumentState) -> bool:
    """Whether a transition is legal."""
    return target in TRANSITIONS[source]


def transition(
    session: Session, doc: Document, target: DocumentState, *, reason: str | None = None
) -> Document:
    """Move a document to a new state, raising if the move is illegal.

    `reason` is required for `failed` — an unexplained failure is
    undiagnosable after the fact, and FR-5 requires the reason be kept.

    The keyword index's denormalised copy of the state is synced here rather
    than at each call site. `chunk_terms` holds a copy so keyword queries can
    filter without joining `documents` (architecture.md 4.3), and a copy that
    some caller forgets to update serves a tombstoned document or hides a live
    one — both invisible in the database. Every state change therefore goes
    through this one function, so the copy cannot drift.
    """
    source = doc.state
    if not can_transition(source, target):
        raise StateTransitionError(doc.doc_id, source.value, target.value)

    if target == DocumentState.FAILED and not reason:
        raise ValueError("a transition to 'failed' requires a reason")

    doc.state = target
    if reason:
        doc.error_reason = reason
    if target == DocumentState.FAILED:
        log.warning("document failed", extra={"doc_id": doc.doc_id, "reason": reason})
    else:
        # Clear any stale reason so a recovered document does not keep
        # displaying the previous failure.
        doc.error_reason = None

    log.info(
        "state transition",
        extra={"doc_id": doc.doc_id, "from": source.value, "to": target.value},
    )
    session.flush()

    # After the flush, so the sync sees the new state rather than the old one.
    # Imported lazily: `keyword` imports models, and a module-level import here
    # would make the dependency visible in both directions.
    from app.ingest.keyword import refresh_keyword_state

    refresh_keyword_state(session, doc.doc_id)
    session.flush()

    return doc


def mark_failed(session: Session, doc: Document, reason: str) -> Document:
    """Move to `failed` and record why.

    Called from the exception handler for any stage. A document already in
    `failed` is left alone rather than re-transitioned, so the first (root cause)
    reason is the one preserved.
    """
    if doc.state == DocumentState.FAILED:
        return doc
    if not can_transition(doc.state, DocumentState.FAILED):
        # Already terminal (deleted, superseded). Nothing to record.
        log.info(
            "ignoring failure for terminal document",
            extra={"doc_id": doc.doc_id, "state": doc.state.value},
        )
        return doc
    return transition(session, doc, DocumentState.FAILED, reason=reason)


def is_retrievable(doc: Document) -> bool:
    """Whether a document may be served as a citation source."""
    return doc.state in RETRIEVABLE_STATES
