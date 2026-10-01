"""State machine: legal transitions and the two invariants it encodes.

`test_state_machine.py` from implementation.md 3.5. The table is asserted
exhaustive and legal, because a transition missing from it fails silently in
production: a document would sit in a transient state with no indication that
its work never finished.
"""

from __future__ import annotations

import itertools

import pytest

from app.core.errors import StateTransitionError
from app.db.models import Document, DocumentState
from app.ingest.states import (
    RETRIEVABLE_STATES,
    TRANSITIONS,
    can_transition,
    is_retrievable,
    mark_failed,
    transition,
)

PIPELINE = [
    DocumentState.PENDING,
    DocumentState.EXTRACTING,
    DocumentState.CHUNKING,
    DocumentState.EMBEDDING,
    DocumentState.INDEXING,
    DocumentState.LIVE,
]


def _doc(**kwargs) -> Document:
    return Document(filename="x.md", mime_type="text/markdown", byte_size=10, **kwargs)


def test_every_state_has_a_transition_entry():
    """No state may be absent from the table — that is how transitions get lost."""
    for state in DocumentState:
        assert state in TRANSITIONS, f"{state} missing from the transition table"


def test_pipeline_path_is_legal():
    for source, target in itertools.pairwise(PIPELINE):
        assert can_transition(source, target), f"{source} -> {target} should be legal"


def test_every_pipeline_state_can_fail():
    """Any stage can fail, so any state on the path must reach `failed`."""
    for state in PIPELINE[:-1]:
        assert can_transition(state, DocumentState.FAILED), (
            f"{state} must be able to fail; otherwise a mid-pipeline crash leaves "
            "the document stuck with no terminal state"
        )


def test_cannot_skip_ahead():
    """Skipping states is illegal.

    Jumping straight to `live` would mark a document indexed before its chunks
    were written, which is the half-indexed document the design forbids.
    """
    assert not can_transition(DocumentState.PENDING, DocumentState.LIVE)
    assert not can_transition(DocumentState.EXTRACTING, DocumentState.INDEXING)
    assert not can_transition(DocumentState.CHUNKING, DocumentState.LIVE)


def test_cannot_move_backwards():
    assert not can_transition(DocumentState.LIVE, DocumentState.CHUNKING)
    assert not can_transition(DocumentState.INDEXING, DocumentState.EMBEDDING)


def test_only_live_is_retrievable():
    """Invariant 1: only `live` is retrievable."""
    assert frozenset({DocumentState.LIVE}) == RETRIEVABLE_STATES
    for state in DocumentState:
        expected = state == DocumentState.LIVE
        assert is_retrievable(_doc(state=state)) is expected, (
            f"{state} retrievable should be {expected}"
        )


def test_deleted_is_terminal():
    assert TRANSITIONS[DocumentState.DELETED] == frozenset()


def test_superseded_cannot_be_re_enabled():
    """A superseded document has been replaced; re-enabling it would serve two
    conflicting versions of the same policy (FR-8)."""
    assert not can_transition(DocumentState.SUPERSEDED, DocumentState.LIVE)


def test_disabled_can_be_re_enabled():
    assert can_transition(DocumentState.DISABLED, DocumentState.LIVE)


def test_failed_can_restart():
    """A failed document can be re-ingested rather than requiring a re-upload."""
    assert can_transition(DocumentState.FAILED, DocumentState.PENDING)


def test_transition_updates_state(session):
    doc = _doc(state=DocumentState.PENDING)
    session.add(doc)
    session.flush()

    transition(session, doc, DocumentState.EXTRACTING)
    assert doc.state == DocumentState.EXTRACTING
    assert doc.error_reason is None


def test_illegal_transition_raises(session):
    doc = _doc(state=DocumentState.PENDING)
    session.add(doc)
    session.flush()

    with pytest.raises(StateTransitionError) as excinfo:
        transition(session, doc, DocumentState.LIVE)
    assert excinfo.value.source == "pending"
    assert excinfo.value.target == "live"
    assert doc.state == DocumentState.PENDING, "illegal transition must not mutate state"


def test_failure_requires_a_reason(session):
    """An unexplained failure is undiagnosable after the fact (FR-5)."""
    doc = _doc(state=DocumentState.EXTRACTING)
    session.add(doc)
    session.flush()

    with pytest.raises(ValueError, match="requires a reason"):
        transition(session, doc, DocumentState.FAILED)


def test_mark_failed_records_reason(session):
    doc = _doc(state=DocumentState.CHUNKING)
    session.add(doc)
    session.flush()

    mark_failed(session, doc, "chunking produced no chunks")
    assert doc.state == DocumentState.FAILED
    assert doc.error_reason == "chunking produced no chunks"


def test_mark_failed_preserves_root_cause(session):
    """A second failure must not overwrite the first reason.

    The first is the root cause; a later one is usually a consequence.
    """
    doc = _doc(state=DocumentState.CHUNKING)
    session.add(doc)
    session.flush()

    mark_failed(session, doc, "root cause: extractor crashed")
    mark_failed(session, doc, "consequence: cleanup also failed")
    assert doc.error_reason == "root cause: extractor crashed"


def test_transition_clears_stale_error_reason(session):
    doc = _doc(state=DocumentState.FAILED, error_reason="earlier failure")
    session.add(doc)
    session.flush()

    transition(session, doc, DocumentState.PENDING)
    assert doc.error_reason is None, "a recovered document must not keep showing the old failure"
