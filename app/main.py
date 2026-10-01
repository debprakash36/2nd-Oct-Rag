"""FastAPI application factory.

Surfaces health, admin document management, streaming chat, conversations,
citations, and feedback. Routers are registered in that order so `/openapi.json`
groups them the way a reader expects: infrastructure, then content, then the user-
facing chat surface and its dependents.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.api import (
    admin_documents,
    chat,
    chunks,
    conversations,
    feedback,
    health,
)
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.guardrails import RateLimitExceeded
from app.core.logging import configure_logging, get_logger
from app.core.middleware import TraceIdMiddleware
from app.db.session import create_all, get_engine, get_session_factory

log = get_logger("app.main")


def _assert_vector_store_ready(settings: Settings) -> None:
    """Fail at startup if retrieval cannot answer, not on the first user request.

    An unusable vector store is otherwise discovered by a user, as a refusal. By
    then the request has already been written, the log row may be half-written, and
    the operator is looking at a "the bot said it couldn't help" bug report rather
    than a configuration error. Raising here names the problem and the fix.

    Skipped when the schema does not exist yet: a database with no documents is a
    fresh install, not a misconfiguration, and refusing to boot would make an empty
    deployment impossible to start.
    """
    from app.retrieval.vector_store import assert_store_usable, build_vector_store

    session = get_session_factory(settings)()
    try:
        assert_store_usable(session, build_vector_store(session, settings))
    except SQLAlchemyError:
        log.warning(
            "skipping vector store check: database schema unavailable",
            extra={"hint": "run migrations, or ingest a corpus"},
        )
    finally:
        session.close()


def _assert_providers_constructible(settings: Settings) -> None:
    """Fail at startup if a selected provider cannot be built.

    Both provider factories are otherwise called lazily, on the first request that
    needs them. For a hosted provider that means a missing API key surfaces as a
    503 on a user's question rather than as a boot failure, which inverts the cost:
    the operator sees a stream of support complaints instead of one clear startup
    error naming the variable to set.

    This constructs each provider and discards it. It deliberately does not make a
    network call -- a boot that depends on a third party being up turns an upstream
    outage into a restart loop, and the health check already surfaces reachability.
    What it catches is configuration: an empty key, an unknown provider name, a
    model/dimension pair that cannot be constructed.
    """
    from app.core.errors import AppError
    from app.providers.embedding import get_embedding_provider
    from app.providers.generation import get_generation_provider

    for name, factory in (
        ("embedding", get_embedding_provider),
        ("generation", get_generation_provider),
    ):
        try:
            factory(settings)
        except AppError as exc:
            # Re-raised rather than logged: an unusable provider means the system
            # cannot answer, and starting anyway only defers the failure.
            raise RuntimeError(
                f"{name} provider is not usable: {exc.detail}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"{name} provider is not usable: {exc}"
            ) from exc


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate configuration and ensure the schema exists on startup."""
    settings = get_settings()
    configure_logging(settings.log_level)
    # Fail fast on a misconfigured deployment rather than on the first request.
    settings.validate_production()
    # Register the session factory with the app's own settings before anything can
    # ask for it. `get_db` deliberately calls `get_session_factory()` with no
    # argument, on the basis that the engine is already bound; this is what binds it.
    # Previously the first caller resolved the engine implicitly from the global
    # settings singleton, which is correct in production and was actively harmful in
    # tests -- the object a test overrides is the FastAPI dependency, not that
    # singleton, so the first caller in a test process bound to whatever
    # DATABASE_URL said.
    get_session_factory(settings)
    if settings.environment in {"local", "test"}:
        create_all(get_engine(settings))
    _assert_providers_constructible(settings)
    _assert_vector_store_ready(settings)
    log.info(
        "startup",
        extra={
            "environment": settings.environment,
            "embedding_dim": settings.embedding_dim,
            "vector_store": settings.vector_store,
        },
    )
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Chatbot",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(TraceIdMiddleware)
    # The web UI is a separate origin, so its XHRs need this. Origins are read from
    # settings rather than hardcoded, and default to localhost only: a misconfigured
    # deployment should refuse browser origins, not allow all of them.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        # Only methods the client actually uses. PUT appears for feedback votes.
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Trace-Id"],
        # No cookies: nothing here is authenticated, and allowing credentials with a
        # wildcard origin would be a vulnerability the day auth lands.
        allow_credentials=False,
        # Preflight responses are cheap and static; a short cache keeps the browser
        # from re-asking on every request.
        max_age=600,
    )

    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        """Return only the user-safe message (NFR-5).

        `exc.detail` stays in the log. It can carry a filesystem path, a provider
        error, or a document id, none of which belong in a response.

        The status comes from the error (`exc.status_code`), not from a blanket 400.
        That distinction is load-bearing for two cases a 400 would break: a client
        polling a deleted conversation would retry forever against a 400, and a rate
        limited client would show an error rather than backing off.
        """
        log.warning("app error", extra={"detail": exc.detail, "error_type": type(exc).__name__})
        headers = {}
        if isinstance(exc, RateLimitExceeded):
            # Retry-After is what lets a well-behaved client stop hammering; without
            # it the client has to guess and most guess immediately.
            headers["Retry-After"] = str(exc.retry_after_seconds)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.user_message},
            headers=headers,
        )

    app.include_router(health.router)
    app.include_router(admin_documents.router)
    app.include_router(chat.router)
    app.include_router(conversations.router)
    app.include_router(chunks.router)
    app.include_router(feedback.router)
    return app


app = create_app()
