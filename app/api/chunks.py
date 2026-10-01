"""Exact-passage lookup for clickable citations (FR-18, FR-20, NFR-9).

Clicking a citation marker must reveal the *exact* passage the claim was drawn from
and its source location. That sentence can be up to `chunk_max_chars` long (1 200 by
default) plus a multi-page breadcrumb, so it does not fit in the SSE `sources` event
without inflating every answer's framing payload. It is fetched on demand instead,
which also means a user who never clicks a citation never pays for it.

This endpoint is not in the architecture's API table. It is the minimum addition
needed to satisfy FR-18's "exact passage" requirement with a payload that does not
degrade the streaming path, and it is called out here rather than added silently.

**Access control caveat.** v1 has no authentication (architecture.md NG5), so this
endpoint cannot check who is asking. It does mirror the chat path's effective policy
by requiring the chunk's document to be `LIVE`, which at least prevents passage
text from a disabled or soft-deleted document leaking through a citation the user
saved earlier. That is parity with `/chat/stream`, not authorization: anyone who can
reach the app can read any live chunk. Do not expose publicly without auth.

The text is returned as a JSON string and rendered by the client as text, never as
HTML (FR-22). Nothing here needs to escape it — that guarantee is made at the render
boundary, not here.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.db.models import Chunk, Document, DocumentState
from app.db.session import get_db

router = APIRouter(prefix="/chunks", tags=["citations"])


class PassageOut(BaseModel):
    """The exact retrieved passage behind one citation marker."""

    chunk_id: str
    document_id: str
    filename: str
    breadcrumb: str | None
    page: int | None
    text: str


@router.get("/{chunk_id}", response_model=PassageOut)
def read_passage(chunk_id: str, session: Session = Depends(get_db)) -> PassageOut:
    """Return the full passage for `chunk_id`.

    404s for a missing chunk *and* for a chunk whose document is no longer live, via
    the same error type. Distinguishing them would tell an unauthenticated caller
    which chunk ids exist, which is a small amount of free reconnaissance for no
    user-visible benefit.
    """
    row = session.execute(
        select(Chunk, Document)
        .join(Document, Document.doc_id == Chunk.doc_id)
        .where(Chunk.chunk_id == chunk_id)
    ).first()
    if row is None:
        raise NotFoundError("chunk")

    chunk, document = row
    if document.state is not DocumentState.LIVE:
        # Logged internally and 404'd externally: a citation that outlived its
        # document is expected, but we want to see how often it happens before
        # deciding it deserves user-facing wording.
        raise NotFoundError("passage for a document that is no longer live")

    return PassageOut(
        chunk_id=chunk.chunk_id,
        document_id=document.doc_id,
        filename=document.filename,
        breadcrumb=chunk.breadcrumb,
        page=chunk.page,
        text=chunk.text,
    )