"""Chunk persistence and embedding (FR-12 prep).

The chunk store is the **authoritative** membership record (architecture.md
4.3). Everything that decides what exists reads `Chunk`; the vector index and
keyword index are derived from it. That is what makes index drift recoverable by
rebuild rather than a permanent correctness bug.

Writes go through this module for the same reason: if chunks were written from
two places, the two writers would eventually disagree about membership, and the
disagreement would be invisible.

Embedding input is `breadcrumb + "\n" + text`, not the stored text. The
breadcrumb restores the chunk's subject at embedding time (FR-11) without
polluting search results or citations with document titles.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import EmbeddingError
from app.core.logging import get_logger
from app.db.models import Chunk, Document
from app.ingest.chunk import ChunkDraft
from app.ingest.keyword import purge_keyword_index, refresh_keyword_state, write_keyword_index
from app.providers.base import EmbeddingProvider

log = get_logger("app.ingest.index")

#: Chunks per embedding request. Bounded so a large document does not produce a
#: single oversized request; the provider is free to batch further.
EMBED_BATCH_SIZE = 64


def embedding_input(breadcrumb: str, text: str) -> str:
    """The string actually sent to the embedding provider.

    Breadcrumb first so the subject leads the representation. Never stored on
    `Chunk.text`.
    """
    if not breadcrumb:
        return text
    return f"{breadcrumb}\n{text}"


def write_chunks(
    session: Session,
    doc: Document,
    drafts: Sequence[ChunkDraft],
    provider: EmbeddingProvider,
    settings: Settings,
) -> list[Chunk]:
    """Persist chunks and their embeddings, replacing any previous set.

    Deleting and re-inserting within one transaction is intentional. A partial
    update is worse than no update: the document would serve a mixture of old and
    new chunks, and a citation could point at text that is no longer in the file.
    Either the whole set lands or the transaction rolls back.

    Returns the persisted chunks. Callers must flush before the document reaches
    `live` so the index is consistent with the state.
    """
    session.execute(delete(Chunk).where(Chunk.doc_id == doc.doc_id))

    chunks: list[Chunk] = []
    for offset in range(0, len(drafts), EMBED_BATCH_SIZE):
        batch = drafts[offset : offset + EMBED_BATCH_SIZE]
        inputs = [embedding_input(d.breadcrumb, d.text) for d in batch]
        vectors = _embed(provider, inputs, settings)
        for draft, vector in zip(batch, vectors, strict=True):
            chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_index=draft.index,
                    text=draft.text,
                    breadcrumb=draft.breadcrumb,
                    section_path=list(draft.section_path),
                    token_count=draft.token_count,
                    char_start=draft.char_start,
                    char_end=draft.char_end,
                    page=draft.page,
                    embedding=vector,
                    embedding_model=settings.embedding_model,
                )
            )

    # Appended through the relationship rather than `session.add_all` so the
    # already-loaded `doc.chunks` collection stays accurate. Adding rows by
    # `doc_id` alone leaves a previously-loaded collection empty, and a caller
    # holding the document would then see zero chunks for a document that has
    # them.
    doc.chunks.extend(chunks)
    session.flush()
    log.info(
        "chunks written",
        extra={"doc_id": doc.doc_id, "chunks": len(chunks), "dim": settings.embedding_dim},
    )
    return chunks


def _embed(
    provider: EmbeddingProvider, inputs: Sequence[str], settings: Settings
) -> list[list[float]]:
    """Call the provider and validate the response shape.

    A length mismatch is raised rather than zipped. `zip(..., strict=True)` in
    the caller would catch it too, but a wrong-width vector would pass that check
    and silently produce meaningless similarity scores — which is exactly the
    failure the pinned `embedding_dim` in config exists to prevent.
    """
    vectors = provider.embed(inputs, model=settings.embedding_model)
    if len(vectors) != len(inputs):
        raise EmbeddingError(
            f"provider returned {len(vectors)} vectors for {len(inputs)} inputs"
        )
    for vector in vectors:
        if len(vector) != settings.embedding_dim:
            raise EmbeddingError(
                f"provider returned dimension {len(vector)}, expected "
                f"{settings.embedding_dim}. The embedding model has probably "
                "changed; existing vectors are invalid and the index must be rebuilt."
            )
    return vectors


def chunk_count(session: Session, doc_id: str) -> int:
    """Number of chunks stored for a document. Used by the admin API (FR-27)."""
    return len(
        session.execute(select(Chunk.chunk_id).where(Chunk.doc_id == doc_id)).all()
    )


def purge_chunks(session: Session, doc_id: str) -> int:
    """Delete a document's chunks. Returns the count removed.

    Used on failure cleanup. Leaving partial chunks behind would let a document
    that never reached `live` contribute passages to an answer, and a
    half-indexed document is worse than a missing one: the system would answer
    confidently from a source the user cannot fully read.
    """
    # Keyword rows cascade from `chunks` on Postgres, but SQLite only does so with
    # foreign keys enabled, so they are removed explicitly. Done before the chunks
    # go, since purging resolves chunk ids by querying `chunks`.
    purged_terms = purge_keyword_index(session, doc_id)

    result = session.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
    session.flush()
    # A DML DELETE returns a CursorResult with `rowcount`; the declared
    # `Result` type is wider. `rowcount` can be -1 when the dialect cannot
    # report it, which is reported as zero rather than as a negative count.
    removed = max(getattr(result, "rowcount", 0) or 0, 0)
    if removed or purged_terms:
        log.info(
            "purged chunks",
            extra={"doc_id": doc_id, "chunks": removed, "keyword_rows": purged_terms},
        )
    return removed


def write_indexes(session: Session, doc: Document) -> int:
    """Write every derived index for a document.

    Called once chunks are persisted but before the document goes `live`, so the
    derived indexes and the authoritative chunk store land in the same
    transaction. A `live` document is by construction one whose indexes are
    already written — there is no window in which retrieval could see a live
    document missing from the keyword index.
    """
    written = write_keyword_index(session, doc.doc_id)
    session.flush()
    log.info("indexes written", extra={"doc_id": doc.doc_id, "keyword_rows": written})
    return written


def finalize_document(
    session: Session, doc: Document, *, indexed: bool = True
) -> Document:
    """Stamp a document as fully indexed and sync index state.

    `indexed=False` marks a document as no longer serving without deleting its
    indexes, so a temporary failure can be reversed without a re-embed.
    """
    doc.indexed_at = dt.datetime.now(dt.UTC) if indexed else None
    refresh_keyword_state(session, doc.doc_id)
    session.flush()
    return doc
