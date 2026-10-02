"""Thumbs feedback on an answer (FR-29, architecture.md 6.1).

One row per answer, updated in place. FR-29 specifies "thumbs up/down"; it does not
specify free text, and free text would be a different feature with a different
privacy story (it would start storing user-authored content outside the redaction
path that FR-33 applies to questions). Not building it.

Storing feedback on the `QueryLog` row rather than a separate table is what makes
the PRD's improvement loop possible: "show me the down-voted queries" is a filter on
`feedback IS NOT NULL AND feedback = 'down'` joined to the per-stage scores already
in the same row. A separate feedback table would need the join anyway.

Clicks are idempotent and reversible by overwriting. A double-click on "up" is not
an error, and neither is switching from up to down — that is someone reconsidering,
not a conflict.

When `API_TOKEN` is set, this endpoint is behind the same bearer gate as the
rest of the API. Empty token still means open, which is what tests need.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.db.models import QueryLog
from app.db.session import get_db

log = get_logger("app.api.feedback")

router = APIRouter(prefix="/feedback", tags=["feedback"])

FeedbackValue = Literal["up", "down", "none"]


class FeedbackIn(BaseModel):
    """Feedback for one answer.

    `query_id` is required in the body rather than in the path so the payload
    matches the other write endpoints' shape and the client sends one object. A
    `PUT /feedback/{query_id}` would be marginally more RESTful and would make the
    idempotent-retry story rely on the HTTP verb rather than on the body, which is
    the property that actually matters here.
    """

    query_id: str = Field(min_length=1, max_length=64)
    #: `none` clears a previous vote. Modelled as a value rather than a `DELETE`
    #: because clearing is a state the user can reach from the same button, and a
    #: UI that cannot undo a click produces support questions.
    value: FeedbackValue


class FeedbackOut(BaseModel):
    query_id: str
    value: str


@router.put("", response_model=FeedbackOut)
def set_feedback(payload: FeedbackIn, session: Session = Depends(get_db)) -> FeedbackOut:
    """Record or clear feedback for one answer (FR-29).

    Uses a single conditional UPDATE rather than a SELECT-then-UPDATE so a
    concurrent second click cannot interleave and write a stale value back. The
    rowcount tells us whether the id existed, which avoids a follow-up query on the
    happy path.
    """
    value = None if payload.value == "none" else payload.value

    result = session.execute(
        update(QueryLog)
        .where(QueryLog.query_id == payload.query_id)
        .values(feedback=value)
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise NotFoundError("answer")

    session.commit()
    log.info(
        "feedback recorded",
        # query_id is logged: it is a correlation key for the improvement loop and
        # is not user-authored content.
        extra={"query_id": payload.query_id, "feedback": value},
    )
    return FeedbackOut(query_id=payload.query_id, value=payload.value)


@router.get("/{query_id}", response_model=FeedbackOut)
def read_feedback(query_id: str, session: Session = Depends(get_db)) -> FeedbackOut:
    """Current feedback for one answer.

    Read back so a page reload restores the thumb state. Without it a reloaded page
    shows both thumbs unselected and the user clicks again, which double-counts in
    any future analysis.
    """
    entry = session.get(QueryLog, query_id)
    if entry is None:
        raise NotFoundError("answer")
    return FeedbackOut(query_id=query_id, value=entry.feedback or "none")


@router.delete("/{query_id}", status_code=status.HTTP_204_NO_CONTENT)
def clear_feedback(query_id: str, session: Session = Depends(get_db)) -> Response:
    """Explicit clear, for clients that prefer a DELETE to `value="none"`.

    404s on an unknown id rather than returning 204 for a no-op. A silently
    successful DELETE hides a client bug — the wrong id being sent, most likely —
    behind a success the UI would render as a cleared thumb.
    """
    result = session.execute(
        update(QueryLog).where(QueryLog.query_id == query_id).values(feedback=None)
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise NotFoundError("answer")
    session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)