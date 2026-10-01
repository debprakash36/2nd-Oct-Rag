"""Health check.

Reports dependency reachability rather than bare liveness. Retrieval cannot
answer without the chunk store, so an endpoint that returns 200 while the
database is unreachable tells a load balancer the service is healthy when it is
not — and a degraded-but-200 response keeps routing users to an instance that
cannot answer (architecture.md 8).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.session import get_db
from app.providers.embedding import get_embedding_provider

router = APIRouter(tags=["health"])


def _check_retrieval_stores(
    session: Session, settings: Settings
) -> tuple[dict[str, str], bool]:
    """Whether the retrieval indexes can actually answer a query.

    architecture.md §8 is explicit that retrieval depends on the vector and keyword
    stores, so a 200 here while either is unreachable would report the service healthy
    when it cannot answer -- the exact condition NFR-2 is measured on.

    Both are probed rather than assumed. The check is a cheap count against the same
    tables the search path reads, so it fails for the reason the request path would
    fail rather than for some unrelated reason.

    Returns the checks and whether retrieval is usable, because "the store answered
    a count query" and "the store can return results" are different questions and
    only the first was previously asked.
    """
    from app.db.models import Chunk, ChunkTerm

    checks: dict[str, str] = {}
    usable = True
    for name, model in (("vector_store", Chunk), ("keyword_index", ChunkTerm)):
        try:
            session.execute(select(func.count()).select_from(model)).scalar_one()
            checks[name] = "ok"
        except SQLAlchemyError as exc:
            # Type only: the message can carry the connection string (NFR-5).
            checks[name] = f"error: {type(exc).__name__}"
            usable = False

    # A reachable store that cannot return results is the failure NFR-2 is measured
    # on, so the count is reported rather than a bare "ok". This is what makes a
    # derived index sitting stale beside the serving store visible: the numbers
    # differ, and a reader does not have to go and count the collection by hand to
    # find out. Empty makes the service unusable, so it drives the 503; merely
    # behind does not, because a derived index is stale by construction after any
    # corpus change (architecture.md 4.3).
    try:
        from app.retrieval.vector_store import build_vector_store, measure_divergence

        divergence = measure_divergence(session, build_vector_store(session, settings))
        checks["vector_index"] = divergence.summary()
        if divergence.is_empty:
            usable = False
    except AppError as exc:
        checks["vector_index"] = f"unavailable: {exc}"
        usable = False
    except RuntimeError as exc:
        # A misconfigured backend (dialect mismatch, unknown name) is a deployment
        # fault, not a dependency outage, and must not read as "index is fine".
        checks["vector_index"] = f"misconfigured: {exc}"
        usable = False

    return checks, usable


@router.get("/health")
def health(
    response: Response,
    session: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Liveness plus dependency checks. 503 when a dependency is down."""
    checks: dict[str, Any] = {}
    healthy = True

    try:
        session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except SQLAlchemyError as exc:
        # The exception type only. The message can contain the connection string,
        # which would leak credentials into a response (NFR-5).
        checks["database"] = f"error: {type(exc).__name__}"
        healthy = False

    try:
        settings.validate_production()
        checks["embedding_provider"] = type(get_embedding_provider(settings)).__name__
    except Exception as exc:
        checks["embedding_provider"] = f"error: {type(exc).__name__}"
        healthy = False

    # Retrieval depends on both indexes (architecture.md §8). Checked separately from
    # the database because a store can be unreachable while SQL is fine -- that is the
    # case where a bare `SELECT 1` reports healthy and every question then fails.
    #
    # Usability comes back from the check rather than being re-derived from the
    # string values: `vector_index` legitimately reports a non-"ok" value
    # ("304/304 live chunks") on a perfectly healthy system, and treating every
    # non-"ok" string as a fault would 503 a working deployment.
    store_checks, stores_usable = _check_retrieval_stores(session, settings)
    checks.update(store_checks)
    if not stores_usable:
        healthy = False

    checks["environment"] = settings.environment
    checks["embedding_dim"] = settings.embedding_dim

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if healthy else "degraded", "checks": checks}
