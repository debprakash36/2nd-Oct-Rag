"""Failure cleanup and tombstone semantics (FR-7).

The invariant under test: a document that did not reach `live` must not leave
retrievable content behind. A half-indexed document is worse than a missing one
— the system would answer confidently from a source the user cannot fully read.
"""

from __future__ import annotations

import pytest

from app.core.errors import NotFoundError
from app.db.models import Chunk, Document, DocumentState
from app.ingest.index import purge_chunks
from app.ingest.tombstone import (
    all_documents,
    disable,
    enable,
    get_document,
    live_documents,
    mark_deleted,
)
from app.ingest.worker import create_document, ingest_document


class _ExplodingProvider:
    """Fails on the second call so the document is live, then broken.

    Models the realistic case: extraction and chunking succeed, then the
    embedding provider fails. The document must not be left holding chunks.
    """

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts, *, model):
        self.calls += 1
        if self.calls > 1:
            from app.core.errors import EmbeddingError

            raise EmbeddingError("provider unavailable")
        return [[0.1] * 64 for _ in texts]


def test_failed_document_leaves_no_chunks(session, object_store, settings_env, sample_md):
    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    from sqlalchemy import select

    # Manually place the document mid-pipeline with chunks present, then fail it.
    doc.state = DocumentState.EMBEDDING
    session.add(Chunk(doc_id=doc.doc_id, chunk_index=0, text="orphan chunk",
                      char_start=0, char_end=13, token_count=2))
    session.commit()

    from app.ingest.states import mark_failed

    mark_failed(session, doc, "simulated failure")
    purge_chunks(session, doc.doc_id)
    session.commit()

    remaining = session.execute(
        select(Chunk).where(Chunk.doc_id == doc.doc_id)
    ).scalars().all()
    assert not remaining, "a failed document must leave no index entries"
    assert doc.state == DocumentState.FAILED


def test_ingest_failure_marks_failed_and_purges(session, object_store, settings_env, sample_md):
    from app.core.errors import EmbeddingError

    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()

    provider = _ExplodingProvider()
    # First call succeeds during a normal ingest; this test focuses on the
    # already-failed path, so force a failure by pre-loading the provider.
    provider.calls = 1

    with pytest.raises(EmbeddingError):
        ingest_document(session, doc.doc_id, object_store=object_store,
                        provider=provider, settings=settings_env)
    session.commit()

    reloaded = session.get(Document, doc.doc_id)
    assert reloaded.state == DocumentState.FAILED
    assert reloaded.error_reason, "a failure must record a reason (FR-5)"
    assert not reloaded.chunks, "a failed document must not leave chunks behind"


def test_unsupported_file_fails_gracefully(session, object_store, provider, settings_env):
    doc = create_document(session, filename="archive.zip", data=b"PK\x03\x04binary",
                          object_store=object_store)
    session.commit()

    from app.core.errors import ExtractionError

    with pytest.raises(ExtractionError):
        ingest_document(session, doc.doc_id, object_store=object_store,
                        provider=provider, settings=settings_env)
    session.commit()

    reloaded = session.get(Document, doc.doc_id)
    assert reloaded.state == DocumentState.FAILED
    assert "unsupported" in (reloaded.error_reason or "").lower()


def test_disable_hides_from_live_but_keeps_chunks(
    session, object_store, provider, settings_env, sample_md
):
    """Disable is a tombstone, not a delete (FR-7)."""
    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    ingest_document(session, doc.doc_id, object_store=object_store,
                    provider=provider, settings=settings_env)
    session.commit()

    chunk_count_before = len(session.get(Document, doc.doc_id).chunks)
    assert chunk_count_before > 0

    disable(session, doc.doc_id)
    session.commit()

    reloaded = session.get(Document, doc.doc_id)
    assert reloaded.state == DocumentState.DISABLED
    assert len(reloaded.chunks) == chunk_count_before, "disable must not remove chunks"
    assert doc.doc_id not in [d.doc_id for d in live_documents(session)]


def test_enable_restores_live(session, object_store, provider, settings_env, sample_md):
    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    ingest_document(session, doc.doc_id, object_store=object_store,
                    provider=provider, settings=settings_env)
    session.commit()

    disable(session, doc.doc_id)
    session.commit()
    enable(session, doc.doc_id)
    session.commit()

    assert session.get(Document, doc.doc_id).state == DocumentState.LIVE


def test_superseded_cannot_be_enabled(session, object_store, provider, settings_env, sample_md):
    """Re-enabling a superseded document would serve two conflicting versions."""
    from app.ingest.tombstone import supersede

    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    ingest_document(session, doc.doc_id, object_store=object_store,
                    provider=provider, settings=settings_env)
    session.commit()

    supersede(session, doc.doc_id)
    session.commit()

    enable(session, doc.doc_id)
    session.commit()
    assert session.get(Document, doc.doc_id).state == DocumentState.SUPERSEDED


def test_deleted_is_hidden_from_get(session, object_store, provider, settings_env, sample_md):
    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    ingest_document(session, doc.doc_id, object_store=object_store,
                    provider=provider, settings=settings_env)
    session.commit()

    mark_deleted(session, doc.doc_id)
    session.commit()

    with pytest.raises(NotFoundError):
        get_document(session, doc.doc_id)
    assert doc.doc_id not in [d.doc_id for d in all_documents(session)]


def test_delete_retains_hash_for_recognition(session, object_store, provider,
                                             settings_env, sample_md):
    """The retained content_hash is what lets a re-upload be recognized."""
    doc = create_document(session, filename="policy.md", data=sample_md,
                          object_store=object_store)
    session.commit()
    ingest_document(session, doc.doc_id, object_store=object_store,
                    provider=provider, settings=settings_env)
    session.commit()
    content_hash = session.get(Document, doc.doc_id).content_hash

    mark_deleted(session, doc.doc_id)
    session.commit()

    assert session.get(Document, doc.doc_id).content_hash == content_hash
