"""Admin API: upload, list, disable, enable, delete (FR-1, FR-2, FR-7, FR-27).

Per-file validation: one bad file in a batch must not discard the others (FR-2),
so each file gets its own status and its own error. A batch response reports what
happened to every file rather than failing the whole request.

Ingestion runs synchronously here. The worker pool and queue land with the async
job infrastructure; the endpoint already returns the per-document id and state,
so swapping the call for an enqueue is a change inside this file, not in its
contract. The `pending` state exists in the state machine from the start so the
client already handles the queued case.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import Document, DocumentState
from app.db.session import get_db, get_session_factory
from app.ingest.chunk import ChunkConfig
from app.ingest.index import chunk_count
from app.ingest.objectstore import LocalObjectStore
from app.ingest.queue import DispatchResult, IngestJob, IngestQueue, InlineQueue
from app.ingest.queue_redis import build_broker_queue
from app.ingest.tombstone import all_documents, disable, enable, get_document, mark_deleted
from app.ingest.validate import validate_upload
from app.ingest.worker import IngestOutcome, create_document, ingest_document
from app.providers.embedding import get_embedding_provider

log = get_logger("app.api.admin")

router = APIRouter(prefix="/admin/documents", tags=["admin"])


def get_object_store(settings: Settings = Depends(get_settings)) -> LocalObjectStore:
    return LocalObjectStore.from_settings(settings)


def build_inline_queue(settings: Settings) -> IngestQueue:
    """Queue that runs the pipeline in-process before returning.

    Owns its own session per job. The request-scoped `get_db` session is not reused
    here: a job outliving the request would hold a closed session, and reusing one
    session for the request and the job would make the two commit boundaries
    ambiguous.
    """

    def _handle(job: IngestJob) -> IngestOutcome:
        session = get_session_factory(settings)()
        try:
            outcome = ingest_document(
                session,
                job.doc_id,
                object_store=LocalObjectStore.from_settings(settings),
                provider=get_embedding_provider(settings),
                settings=settings,
                chunk_config=ChunkConfig(),
            )
            session.commit()
            return outcome
        finally:
            session.close()

    return InlineQueue(_handle)


def get_ingest_queue(settings: Settings = Depends(get_settings)) -> IngestQueue:
    """Return the configured ingestion queue.

    With `INGEST_QUEUE_ENABLED` off — the local and test default — the job runs
    inline, so the pipeline completes before the response and `state` is already
    `live` when the caller sees it. When a broker is enabled, the job is handed to
    it and the document stays `pending` until a worker picks it up; `queued` in
    the response says which happened, rather than leaving the caller to infer it
    from a state value.
    """
    if settings.ingest_queue_enabled and settings.ingest_queue_broker_url:
        try:
            return build_broker_queue(settings)
        except Exception as exc:
            log.error("broker queue unavailable", extra={"reason": str(exc)})
            if settings.ingest_queue_required:
                # Accepting an upload into an undeliverable queue leaves a
                # `pending` row that never goes live, which looks to a user
                # exactly like a lost file. Refusing is the honest outcome.
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    "The ingestion queue is unavailable. No files were stored.",
                ) from exc
            # Not required: degrade to inline. Ingestion is slower and bound to
            # the request, but uploads still work.
            log.warning("ingestion queue degraded to inline")
            return build_inline_queue(settings)

    return build_inline_queue(settings)


class DocumentOut(BaseModel):
    """A document as the admin UI sees it (FR-27).

    `chunk_count` is included because "how much did we actually index" is the
    first question when a document looks wrong, and counting on demand would
    make the list view N+1.
    """

    doc_id: str
    filename: str
    mime_type: str
    state: DocumentState
    version: int
    byte_size: int
    page_count: int | None
    chunk_count: int
    content_hash: str | None
    duplicate_of: str | None
    error_reason: str | None
    acl_tags: list[str] = Field(default_factory=list)
    uploaded_at: dt.datetime
    indexed_at: dt.datetime | None

    @classmethod
    def from_row(cls, doc: Document, chunks: int) -> DocumentOut:
        return cls(
            doc_id=doc.doc_id,
            filename=doc.filename,
            mime_type=doc.mime_type,
            state=doc.state,
            version=doc.version,
            byte_size=doc.byte_size,
            page_count=doc.page_count,
            chunk_count=chunks,
            content_hash=doc.content_hash,
            duplicate_of=doc.duplicate_of,
            error_reason=doc.error_reason,
            acl_tags=list(doc.acl_tags or []),
            uploaded_at=doc.uploaded_at,
            indexed_at=doc.indexed_at,
        )


class UploadResult(BaseModel):
    """Per-file outcome. One of these per uploaded file.

    `error` means the file was not ingested. `warning` means it was, but an admin
    should look — currently only duplicates. Keeping them separate is what stops a
    successfully-indexed duplicate from being reported as a rejected upload.
    """

    filename: str
    doc_id: str | None = None
    state: DocumentState | None = None
    chunks: int = 0
    duplicate_of: str | None = None
    #: True when the document was handed to the ingestion queue. False together with
    #: a populated `error` means stored but never processed.
    queued: bool = False
    warning: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    results: list[UploadResult]
    accepted: int
    rejected: int


@router.get("", response_model=list[DocumentOut])
def list_documents(
    include_tombstones: bool = True,
    session: Session = Depends(get_db),
) -> list[DocumentOut]:
    """List documents with state and chunk counts (FR-27)."""
    docs = all_documents(session)
    if not include_tombstones:
        docs = [d for d in docs if d.state == DocumentState.LIVE]
    return [DocumentOut.from_row(d, chunk_count(session, d.doc_id)) for d in docs]


@router.get("/{doc_id}", response_model=DocumentOut)
def get_one(doc_id: str, session: Session = Depends(get_db)) -> DocumentOut:
    """Fetch a single document's ingestion state."""
    doc = get_document(session, doc_id)
    return DocumentOut.from_row(doc, chunk_count(session, doc_id))


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_documents(
    files: list[UploadFile] = File(...),
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    store: LocalObjectStore = Depends(get_object_store),
    queue: IngestQueue = Depends(get_ingest_queue),
) -> UploadResponse:
    """Upload one or more documents and ingest them (FR-1, FR-2).

    Per-file isolation is the point: a single unsupported or oversized file is
    reported as that file's error and the rest of the batch still ingests.
    """
    results: list[UploadResult] = []

    for upload in files:
        try:
            filename, _extension = validate_upload(upload, settings)
        except AppError as exc:
            # Validation failure: this file only. The batch continues.
            results.append(
                UploadResult(
                    filename=upload.filename or "unknown", error=exc.user_message
                )
            )
            continue

        data = upload.file.read()
        try:
            doc = create_document(
                session, filename=filename, data=data, object_store=store
            )
            session.commit()
        except AppError as exc:
            session.rollback()
            results.append(UploadResult(filename=filename, error=exc.user_message))
            continue

        try:
            dispatch = queue.enqueue(IngestJob(doc_id=doc.doc_id))
        except Exception as exc:
            # The `IngestQueue` contract says `enqueue` must not raise for broker
            # problems, but the document row is already committed here, so an
            # exception escaping would turn a stored upload into a 500 — inviting
            # a retry that duplicates the file while the user learns nothing about
            # whether the first attempt landed.
            log.exception("ingest dispatch raised", extra={"doc_id": doc.doc_id})
            dispatch = DispatchResult(
                doc_id=doc.doc_id, queued=False, error=f"{type(exc).__name__}: {exc}"
            )
        session.expire_all()

        refreshed = session.get(Document, doc.doc_id)
        state = refreshed.state if refreshed is not None else DocumentState.PENDING
        chunks = chunk_count(session, doc.doc_id) if state == DocumentState.LIVE else 0

        if not dispatch.queued:
            # The job did not run. The document row exists and is not live, so
            # this file was accepted but will never be served — reported as a
            # failure rather than a success with a `pending` state nobody follows.
            log.error(
                "ingest dispatch failed",
                extra={"doc_id": doc.doc_id, "reason": dispatch.error},
            )
            results.append(
                UploadResult(
                    filename=filename,
                    doc_id=doc.doc_id,
                    state=state,
                    error="This file was stored but could not be queued for "
                    "processing. Nothing was indexed.",
                )
            )
            continue

        duplicate_of = refreshed.duplicate_of if refreshed is not None else None
        results.append(
            UploadResult(
                filename=filename,
                doc_id=doc.doc_id,
                state=state,
                chunks=chunks,
                duplicate_of=duplicate_of,
                queued=dispatch.queued,
                # A duplicate is a flag, not a failure. It is still ingested
                # and still live, so it does not belong in `error` — putting it
                # there would report a successful ingest as rejected (FR-4).
                warning=(
                    "This content already exists in the corpus; an admin "
                    "should decide which copy to keep."
                    if duplicate_of
                    else None
                ),
            )
        )

    accepted = sum(1 for r in results if r.error is None)
    return UploadResponse(
        results=results, accepted=accepted, rejected=len(results) - accepted
    )


@router.post("/{doc_id}/disable", response_model=DocumentOut)
def disable_document(doc_id: str, session: Session = Depends(get_db)) -> DocumentOut:
    """Stop serving a document without removing it (FR-7)."""
    try:
        doc = disable(session, doc_id)
        session.commit()
    except AppError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.user_message) from exc
    return DocumentOut.from_row(doc, chunk_count(session, doc_id))


@router.post("/{doc_id}/enable", response_model=DocumentOut)
def enable_document(doc_id: str, session: Session = Depends(get_db)) -> DocumentOut:
    """Resume serving a disabled document (FR-7)."""
    try:
        doc = enable(session, doc_id)
        session.commit()
    except AppError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.user_message) from exc
    return DocumentOut.from_row(doc, chunk_count(session, doc_id))


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, session: Session = Depends(get_db)) -> None:
    """Tombstone a document as deleted (FR-7).

    Chunks are retained: physical removal is a separate background operation, and
    the retained `content_hash` is what lets a later re-upload of the same content
    be recognized rather than silently duplicated.
    """
    try:
        mark_deleted(session, doc_id)
        session.commit()
    except AppError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, exc.user_message) from exc
