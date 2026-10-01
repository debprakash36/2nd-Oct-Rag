"""Idempotency under queue redelivery (FR-4, FR-8, implementation.md 3.5).

Queue delivery is at least once, so processing the same document three times must
converge to exactly one document, one chunk set, and one vector set. The failure
this prevents: a redelivered message embeds the same chunks twice, and retrieval
returns the same passage twice in a single answer.
"""

from __future__ import annotations

from sqlalchemy import func, select

from app.db.models import Chunk, Document, DocumentState
from app.ingest.dedupe import HashVerdict, check_content_hash, compute_content_hash
from app.ingest.worker import create_document, ingest_document


def _ingest(session, object_store, provider, settings, data: bytes, filename: str = "policy.md"):
    doc = create_document(session, filename=filename, data=data, object_store=object_store)
    session.commit()
    ingest_document(
        session, doc.doc_id, object_store=object_store, provider=provider, settings=settings
    )
    session.commit()
    return doc.doc_id


def test_same_document_processed_three_times_converges(
    session, object_store, provider, settings_env, sample_md
):
    """The 3x test from implementation.md 3.5.

    Runs the pipeline three times over one document row — the exact shape of an
    at-least-once queue redelivering a job — and asserts nothing accumulates.
    """
    doc_id = _ingest(session, object_store, provider, settings_env, sample_md)

    first_chunks = session.execute(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
    ).scalars().all()
    first_vectors = [c.embedding for c in first_chunks]

    for _ in range(2):
        outcome = ingest_document(
            session, doc_id, object_store=object_store, provider=provider,
            settings=settings_env,
        )
        session.commit()
        assert outcome.state == DocumentState.LIVE

    doc_count = session.execute(
        select(func.count()).select_from(Document).where(Document.doc_id == doc_id)
    ).scalar_one()
    assert doc_count == 1, "redelivery must not create a second document row"

    chunks = session.execute(
        select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
    ).scalars().all()
    assert len(chunks) == len(first_chunks), "redelivery must not duplicate chunks"
    assert [c.embedding for c in chunks] == first_vectors, "vectors must not be re-embedded"

    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(set(indices)), "chunk indices must stay unique per document"

    assert session.get(Document, doc_id).state == DocumentState.LIVE


def test_second_upload_of_identical_content_is_flagged_duplicate(
    session, object_store, provider, settings_env, sample_md
):
    """Different file, identical content (FR-4).

    Recorded and kept, but not indexed. Silently dropping it would look like data
    loss from the uploader's side and the admin list would not show what arrived;
    indexing it produced eleven identical LIVE documents, all retrievable, so one
    query returned the same passage eleven times. An admin can promote it from the
    admin surface if the original is removed.
    """
    first_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")
    second_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy-copy.md")

    first = session.get(Document, first_id)
    second = session.get(Document, second_id)

    assert first.doc_id != second.doc_id
    assert second.content_hash == first.content_hash
    assert second.duplicate_of == first.doc_id, "duplicate must be recorded, not dropped"
    assert (
        second.state == DocumentState.DUPLICATE
    ), "a duplicate must not go live; it would be served alongside the original"
    # The original is untouched -- flagging a duplicate must not disturb what works.
    assert first.state == DocumentState.LIVE


def test_a_duplicate_is_not_retrievable(
    session, object_store, provider, settings_env, sample_md
):
    """The whole point of the DUPLICATE state: excluded from every search path.

    `LIVE` is the predicate both the SQL scan and the Chroma projection filter on, so
    this asserts the end-to-end consequence rather than just the state value -- a
    duplicate that is merely labelled but still returned would leave the original
    defect in place while looking fixed.
    """
    from app.retrieval.types import AccessFilter
    from app.retrieval.vector_store import SqliteVectorStore

    first_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")
    second_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy-copy.md")

    store = SqliteVectorStore(session)
    query = provider.embed(["refund policy"], model=settings_env.embedding_model)[0]

    found = store.search(query, k=50, access=AccessFilter())
    found_doc_ids = {c.doc_id for c in found}

    assert first_id in found_doc_ids, "the original must still be retrievable"
    assert second_id not in found_doc_ids, (
        "a duplicate reached retrieval: the same passage would be returned once per "
        "upload, inflating sources and spending the context budget on copies"
    )


def test_a_duplicate_writes_no_chunks_and_no_keyword_terms(
    session, object_store, provider, settings_env, sample_md
):
    """Not indexed at all, so there is nothing for either index to serve.

    The keyword index carries a denormalised copy of the state, so a duplicate that
    was written and then marked would still be filterable one way and not the other.
    Asserting the absence of rows closes that gap.
    """
    from app.db.models import Chunk, ChunkTerm

    _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")
    second_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy-copy.md")

    assert session.query(Chunk).filter(Chunk.doc_id == second_id).count() == 0
    # Vacuously true if the document wrote no chunks at all, which the assertion
    # above already covers -- kept as an independent check that the keyword index
    # has no rows the document could be served through.
    assert (
        session.query(ChunkTerm)
        .join(Chunk, Chunk.chunk_id == ChunkTerm.chunk_id)
        .filter(Chunk.doc_id == second_id)
        .count()
        == 0
    )


def test_duplicate_state_is_actually_assigned():
    """`DocumentState.DUPLICATE` existed in the enum and was set by nothing.

    An unreachable enum member is the kind of thing that rots silently: the code read
    as though duplicates were handled, and eleven LIVE copies proved otherwise. This
    asserts reachability from the enum's own transition table.
    """
    from app.ingest.states import TRANSITIONS

    targets = {t for allowed in TRANSITIONS.values() for t in allowed}
    assert DocumentState.DUPLICATE in targets, (
        "no state can transition to DUPLICATE, so the duplicate branch in "
        "ingest_document could not have been running"
    )
    assert DocumentState.DUPLICATE in TRANSITIONS[DocumentState.CHUNKING]


def test_hash_is_content_based_not_byte_based():
    """The documented reason for hashing chunked content (architecture.md 4.4)."""
    a = compute_content_hash(["hello world", "second chunk"])
    b = compute_content_hash(["hello world", "second chunk"])
    c = compute_content_hash(["different", "text entirely"])
    assert a == b, "identical content must hash identically"
    assert a != c


def test_hash_resists_boundary_ambiguity():
    """`["ab","c"]` and `["a","bc"]` must not collide.

    A naive `''.join()` hashes both to the same value, and two different
    documents would be treated as one — a silent corpus corruption.
    """
    assert compute_content_hash(["ab", "c"]) != compute_content_hash(["a", "bc"])
    assert compute_content_hash(["a", "b"]) != compute_content_hash(["ab"])


def test_same_document_re_upload_is_not_a_duplicate(session, object_store, provider,
                                                    settings_env, sample_md):
    """A document is not a duplicate of itself.

    The verdict for a document matching its own content is SAME_DOCUMENT, not
    DUPLICATE. Getting this wrong would make a redelivered job record itself as
    a duplicate on every attempt.
    """
    doc_id = _ingest(session, object_store, provider, settings_env, sample_md)
    doc = session.get(Document, doc_id)
    assert doc.content_hash is not None

    verdict = check_content_hash(session, doc.content_hash, doc_id)
    assert verdict.verdict == HashVerdict.SAME_DOCUMENT
    assert verdict.existing_doc_id == doc_id


def test_unknown_hash_is_new(session):
    verdict = check_content_hash(session, "0" * 64, "nonexistent")
    assert verdict.verdict == HashVerdict.NEW
    assert verdict.existing_doc_id is None


def test_tombstoned_copy_is_not_a_live_duplicate(session, object_store, provider,
                                                 settings_env, sample_md):
    """Re-uploading content whose other copy was deleted is a NEW document.

    Otherwise a deleted document would permanently block its own re-upload.
    """
    first_id = _ingest(session, object_store, provider, settings_env, sample_md, "policy.md")
    session.get(Document, first_id).state = DocumentState.DELETED
    session.commit()

    first = session.get(Document, first_id)
    verdict = check_content_hash(session, first.content_hash, "a-different-doc")
    assert verdict.verdict == HashVerdict.NEW
