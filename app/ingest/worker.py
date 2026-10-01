"""The ingestion worker (FR-5, G4).

Runs the pipeline for one document:

    pending -> extracting -> chunking -> embedding -> indexing -> live

Idempotency under redelivery
----------------------------
Queue delivery is at least once, so this function must converge, not accumulate.
A redelivered message that ran to completion is recognized by the content hash
and exits early. A redelivered message interrupted mid-pipeline restarts from
`pending` after cleanup, because `write_chunks` replaces the chunk set inside one
transaction — a partial chunk set is never observable.

The failure path purges chunks rather than marking them unretrievable. A
half-indexed document is worse than a missing one: the system would answer
confidently from a source the user cannot fully read, with no indication that
anything went wrong.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError, ValidationError
from app.core.logging import get_logger
from app.db.models import Document, DocumentState
from app.ingest.breadcrumb import detect_headings
from app.ingest.chunk import ChunkConfig, chunk_document
from app.ingest.dedupe import HashVerdict, check_content_hash, compute_content_hash
from app.ingest.index import finalize_document, purge_chunks, write_chunks, write_indexes
from app.ingest.objectstore import LocalObjectStore
from app.ingest.sandbox import run_sandboxed
from app.ingest.states import mark_failed, transition
from app.ingest.validate import guess_mime
from app.providers.base import EmbeddingProvider

log = get_logger("app.ingest.worker")


@dataclass
class IngestOutcome:
    """What happened to one document."""

    doc_id: str
    state: DocumentState
    chunks: int = 0
    duplicate_of: str | None = None
    detail: str = ""


def create_document(
    session: Session,
    *,
    filename: str,
    data: bytes,
    object_store: LocalObjectStore,
    acl_tags: list[str] | None = None,
) -> Document:
    """Register an upload and store its bytes. Does not run the pipeline.

    Separate from `ingest_document` so a queued job operates on a row that
    already exists, which is what makes a redelivered job safe: the document is
    looked up by id, never re-created.
    """
    doc = Document(
        filename=filename,
        mime_type=guess_mime(filename),
        byte_size=len(data),
        state=DocumentState.PENDING,
        acl_tags=acl_tags or [],
        uploaded_at=dt.datetime.now(dt.UTC),
    )
    session.add(doc)
    session.flush()
    doc.object_key = object_store.new_key(doc.doc_id, filename)
    object_store.put(doc.object_key, data)
    session.flush()
    # `doc_name`, not `filename`: `filename` is a reserved LogRecord attribute
    # and `extra` raises on a collision.
    log.info("document registered", extra={"doc_id": doc.doc_id, "doc_name": filename})
    return doc


def ingest_document(
    session: Session,
    doc_id: str,
    *,
    object_store: LocalObjectStore,
    provider: EmbeddingProvider,
    settings: Settings | None = None,
    chunk_config: ChunkConfig | None = None,
) -> IngestOutcome:
    """Run the full pipeline for one document.

    Any exception marks the document `failed` with a reason and purges its
    chunks, then re-raises. Re-raising matters: the worker must not report
    success for a document that did not make it, and a queued job that swallows
    an error is invisible until someone notices the document is missing.
    """
    settings = settings or get_settings()
    chunk_config = chunk_config or ChunkConfig()

    doc = session.get(Document, doc_id)
    if doc is None:
        raise ValidationError(
            f"document {doc_id} not found",
            user_message="This document no longer exists.",
        )

    if doc.state == DocumentState.LIVE:
        # Already complete. A redelivered job landing here is the common case
        # and must be a no-op, not a re-embed.
        log.info("already live, skipping", extra={"doc_id": doc_id})
        return IngestOutcome(doc_id, DocumentState.LIVE, detail="already indexed")

    if doc.state in {DocumentState.DELETED, DocumentState.SUPERSEDED}:
        return IngestOutcome(doc_id, doc.state, detail="tombstoned, not ingested")

    if doc.state == DocumentState.DISABLED:
        # A disabled document that a stale or redelivered job targets must not be
        # resurrected. Retrying a disable by re-ingesting would silently undo the
        # admin's decision (FR-7), and the job cannot know it was fired before the
        # disable. Only an explicit re-ingest, which resets state through the
        # transition table, may bring a disabled document back.
        log.info(
            "document disabled, skipping ingest", extra={"doc_id": doc_id}
        )
        return IngestOutcome(doc_id, doc.state, detail="disabled, not ingested")

    try:
        return _run_pipeline(
            session, doc, object_store, provider, settings, chunk_config
        )
    except AppError as exc:
        _fail(session, doc, exc)
        raise
    except Exception as exc:
        _fail(session, doc, AppError(f"{type(exc).__name__}: {exc}"))
        raise


def _run_pipeline(
    session: Session,
    doc: Document,
    object_store: LocalObjectStore,
    provider: EmbeddingProvider,
    settings: Settings,
    chunk_config: ChunkConfig,
) -> IngestOutcome:
    if doc.state != DocumentState.PENDING:
        # Re-ingesting a failed or disabled document. Reset so the state machine
        # sees a legal transition, and clear chunks left by the previous attempt.
        transition(session, doc, DocumentState.PENDING)
        purge_chunks(session, doc.doc_id)
        doc.error_reason = None

    # --- extract (sandboxed) -------------------------------------------------
    transition(session, doc, DocumentState.EXTRACTING)
    if doc.object_key is None:
        raise ValidationError(
            f"document {doc.doc_id} has no stored object",
            user_message="This document's file is missing. Please re-upload it.",
        )
    raw = object_store.get(doc.object_key)
    extraction = run_sandboxed(
        "extract",
        data=raw,
        filename=doc.filename,
        settings=settings,
    )
    # One canonical cleaned text, persisted. Every offset below is relative to
    # exactly this string.
    doc.cleaned_text = extraction.text
    doc.page_count = extraction.page_count
    session.flush()

    # --- chunk ---------------------------------------------------------------
    transition(session, doc, DocumentState.CHUNKING)
    title = _doc_title(doc.filename)
    drafts = chunk_document(
        extraction.text,
        title,
        headings=detect_headings(extraction.text),
        config=chunk_config,
    )

    # --- dedupe --------------------------------------------------------------
    # Hashed on chunked content, not file bytes: re-saving a PDF changes the
    # bytes without changing the text, and byte hashing would re-embed an
    # identical document while missing the real duplicate case.
    content_hash = compute_content_hash([d.text for d in drafts])

    # Checked before the hash is stored on this row. Assigning it first would put
    # the document into the result set, where its own hash self-matches and every
    # upload looks unchanged to itself.
    verdict = check_content_hash(session, content_hash, doc.doc_id)
    doc.content_hash = content_hash
    session.flush()

    if verdict.verdict == HashVerdict.SAME_DOCUMENT:
        # The chunk set is already correct for this content. Indexes are
        # re-derived rather than assumed present: a prior run may have gone live
        # before the index write, and assuming would leave the keyword index
        # permanently empty for a document that is being served.
        transition(session, doc, DocumentState.INDEXING)
        write_indexes(session, doc)
        finalize_document(session, doc)
        transition(session, doc, DocumentState.LIVE)
        return IngestOutcome(
            doc.doc_id, DocumentState.LIVE, len(drafts), detail="unchanged content"
        )
    if verdict.verdict == HashVerdict.DUPLICATE:
        # Recorded, not rejected: the upload is acknowledged and the admin decides
        # (FR-4). Silently dropping it would look like data loss from the uploader's
        # side, and the admin list needs to show what arrived. The row and its
        # `duplicate_of` pointer are kept so an admin can promote it later.
        #
        # But it must not also be *served*. Falling through to embed/index/live here
        # is what produced eleven identical `policy.md` documents, all LIVE, all
        # retrievable: a query matching that content returned the same passage eleven
        # times, inflating the sources panel and spending the context budget on
        # copies. Flagging is not the same as neutralising -- nothing downstream read
        # the flag, and `DocumentState.DUPLICATE` was never assigned by any code path.
        #
        # Retrieval filters on `LIVE` in both the SQL scan and the Chroma projection,
        # so stopping here excludes the document from every search path without a
        # single change downstream.
        doc.duplicate_of = verdict.existing_doc_id
        log.warning(
            "duplicate content detected",
            extra={"doc_id": doc.doc_id, "duplicate_of": verdict.existing_doc_id},
        )
        transition(session, doc, DocumentState.DUPLICATE)
        session.commit()
        return IngestOutcome(
            doc.doc_id,
            DocumentState.DUPLICATE,
            duplicate_of=verdict.existing_doc_id,
            detail=f"duplicate of {verdict.existing_doc_id}; not indexed",
        )
    session.flush()

    # --- embed and persist ---------------------------------------------------
    transition(session, doc, DocumentState.EMBEDDING)
    chunks = write_chunks(session, doc, drafts, provider, settings)

    # --- index ---------------------------------------------------------------
    transition(session, doc, DocumentState.INDEXING)
    # Derived indexes are written while the document is still `indexing`, in the
    # same transaction as the chunks. Going `live` is therefore the commit point
    # at which everything is consistent — there is no state in which retrieval
    # can serve a live document whose keyword index is missing.
    write_indexes(session, doc)
    finalize_document(session, doc)
    transition(session, doc, DocumentState.LIVE)
    log.info(
        "document live",
        extra={"doc_id": doc.doc_id, "chunks": len(chunks), "duplicate_of": doc.duplicate_of},
    )
    return IngestOutcome(
        doc.doc_id, DocumentState.LIVE, len(chunks), duplicate_of=doc.duplicate_of
    )


def _fail(session: Session, doc: Document, exc: AppError) -> None:
    """Mark failed and purge chunks, in one transaction.

    Purging rather than flagging is deliberate. A document holding partial
    chunks that are still technically present would let retrieval serve passages
    from a source the user cannot fully read — an answer that looks correct and
    is not. Better to have no document than a broken one.
    """
    session.rollback()
    reloaded = session.get(Document, doc.doc_id)
    if reloaded is None:
        return
    purge_chunks(session, reloaded.doc_id)
    mark_failed(session, reloaded, str(exc))
    session.commit()


def _doc_title(filename: str) -> str:
    """Derive a human title from a filename for the breadcrumb (FR-11)."""
    stem = filename.rsplit(".", 1)[0]
    return stem.replace("_", " ").replace("-", " ").strip() or filename
