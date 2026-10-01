"""Keyword index tests.

The property under test is that the keyword index is *derived* — losing it loses
no data — and that a document's tombstone state is enforced on the keyword path
without a join back to `documents`.
"""

from __future__ import annotations

from sqlalchemy import func, select

from app.db.models import Chunk, ChunkTerm, Document, DocumentState
from app.ingest.keyword import (
    keyword_lookup,
    purge_keyword_index,
    tokenize,
    write_keyword_index,
)


def _chunk_ids(session, doc_id: str) -> list[str]:
    return list(session.execute(select(Chunk.chunk_id).where(Chunk.doc_id == doc_id)).scalars())


def test_index_is_derived_from_chunks(session, live_doc, object_store):
    """A keyword row can always be reconstructed from the chunk it points at."""
    write_keyword_index(session, live_doc.doc_id)
    session.commit()

    rows = session.execute(
        select(ChunkTerm).where(ChunkTerm.state == DocumentState.LIVE)
    ).scalars().all()
    assert rows, "expected keyword rows for a live document"

    # Every row resolves to a chunk whose text actually contains the term.
    for row in rows[:20]:
        chunk = session.get(Chunk, row.chunk_id)
        assert chunk is not None
        assert row.term in chunk.text.lower()


def test_lookup_finds_live_content(session, live_doc):
    """The obvious use case: a term from a document finds its own chunks."""
    write_keyword_index(session, live_doc.doc_id)
    session.commit()

    results = keyword_lookup(session, ["refund"])
    assert results, "a term present in the document must be findable"
    assert live_doc.doc_id in {r.doc_id for r in results}


def test_lookup_is_case_insensitive(session, live_doc):
    """Uppercase query terms must match; a user typing in the search box will."""
    write_keyword_index(session, live_doc.doc_id)
    session.commit()

    lower = keyword_lookup(session, ["refund"])
    upper = keyword_lookup(session, ["REFUND"])
    assert {r.chunk_id for r in lower} == {r.chunk_id for r in upper}


def test_disabled_document_is_not_keyword_retrievable(session, live_doc):
    """FR-7: disable must stop the keyword path, not only the vector path.

    This is the failure a join-free index invites. The state is denormalised onto
    the term rows so the filter can apply in the scan, which is only correct if
    the denormalised copy is updated on transition.
    """
    from app.ingest.tombstone import disable

    write_keyword_index(session, live_doc.doc_id)
    session.commit()

    assert keyword_lookup(session, ["refund"]), "precondition: findable while live"

    disable(session, live_doc.doc_id)
    session.commit()

    assert not keyword_lookup(session, ["refund"]), (
        "a disabled document must not be served by keyword search"
    )

    # The rows are retained, not deleted: a tombstone is reversible (FR-7).
    remaining = session.execute(
        select(func.count()).select_from(ChunkTerm)
    ).scalar_one()
    assert remaining > 0, "disable is a tombstone, not a delete"


def test_re_enabled_document_becomes_findable_again(session, live_doc):
    """The reverse transition must restore the index, not just the document row."""
    from app.ingest.tombstone import disable, enable

    write_keyword_index(session, live_doc.doc_id)
    session.commit()

    disable(session, live_doc.doc_id)
    session.commit()
    enable(session, live_doc.doc_id)
    session.commit()

    assert keyword_lookup(session, ["refund"]), "re-enabling must restore retrievability"


def test_deleted_document_is_not_keyword_retrievable(session, live_doc):
    from app.ingest.tombstone import mark_deleted

    write_keyword_index(session, live_doc.doc_id)
    session.commit()
    mark_deleted(session, live_doc.doc_id)
    session.commit()

    assert not keyword_lookup(session, ["refund"])


def test_purge_removes_only_the_targeted_document(session, live_doc, second_live_doc):
    """Purging one document must not disturb another document's index."""
    write_keyword_index(session, live_doc.doc_id)
    write_keyword_index(session, second_live_doc.doc_id)
    session.commit()

    purged = purge_keyword_index(session, live_doc.doc_id)
    session.commit()

    assert purged > 0
    surviving = {r.chunk_id for r in keyword_lookup(session, ["policy", "refund", "shipping"])}
    assert not surviving & set(_chunk_ids(session, live_doc.doc_id)), (
        "the purged document's chunks must no longer be keyword-retrievable"
    )
    assert any(
        r.doc_id == second_live_doc.doc_id for r in keyword_lookup(session, ["policy"])
    ), "the other document's index must be untouched"


def test_rebuild_is_idempotent(session, live_doc):
    """Reindexing the same document must not accumulate duplicate rows."""
    write_keyword_index(session, live_doc.doc_id)
    session.commit()
    first = session.execute(
        select(func.count()).select_from(ChunkTerm)
    ).scalar_one()

    write_keyword_index(session, live_doc.doc_id)
    session.commit()
    second = session.execute(
        select(func.count()).select_from(ChunkTerm)
    ).scalar_one()

    assert first == second, "a rebuild must replace, not accumulate"


def test_tokenizer_drops_single_character_noise():
    """One-character tokens are noise that would match nearly everything."""
    assert "a" not in tokenize("a cat and a dog")
    assert "cat" in tokenize("a cat and a dog")


def test_tokenizer_keeps_numbers():
    """Numbers carry meaning in policies ("30 days", "2 business days")."""
    assert "30" in tokenize("within 30 days")


def test_index_state_matches_document_state_after_ingest(session, live_doc):
    """The denormalised state must match the document, or the index is a dead end.

    The rows are written while the document is still `indexing`, so a copy taken
    at write time records `indexing` rather than `live` — and `keyword_lookup`
    filters on `live`. That makes every freshly ingested document permanently
    unfindable by keyword search while looking perfectly healthy in the database.
    """
    session.expire_all()
    reloaded = session.get(Document, live_doc.doc_id)

    term_states = set(
        session.execute(select(ChunkTerm.state).distinct()).scalars().all()
    )
    assert reloaded is not None
    assert reloaded.state == DocumentState.LIVE
    assert term_states == {DocumentState.LIVE}, (
        f"keyword rows recorded {term_states} but the document is live; the "
        "state copy is written before the document reaches live"
    )
    assert keyword_lookup(session, ["refund"]), (
        "a freshly ingested live document must be immediately keyword-retrievable"
    )