"""Engine and session management.

The engine is created lazily and cached so importing this module has no side
effects — a test that points `DATABASE_URL` at a temp file must not be defeated
by an engine bound at import time.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings
from app.db.models import Base

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def get_engine(settings: Settings | None = None) -> Engine:
    """Return the process-wide engine, creating it on first use."""
    global _engine
    if _engine is None:
        s = settings or get_settings()
        connect_args = {"check_same_thread": False} if s.database_url.startswith("sqlite") else {}
        _engine = create_engine(s.database_url, future=True, connect_args=connect_args)
    return _engine


def get_session_factory(settings: Settings | None = None) -> sessionmaker[Session]:
    """Return the process-wide session factory.

    `settings` may be omitted only while an engine is already cached. Falling
    through to `get_settings()` when one is not is how the test suite came to
    write into the real database: the module-level singleton is not the object the
    `client` fixture overrides, so a no-argument call resolved to
    `sqlite:///./rag.db` and left 64-wide embeddings in a corpus that production
    reads at 384 -- which then failed every retrieval query, because one
    wrong-width row aborts the whole parallel search stage.

    Raising is the right failure. A caller reaching for the global factory before
    an engine is registered is writing somewhere it did not choose, and the
    alternative is silently corrupting a file no assertion covers.
    """
    global _session_factory
    if _session_factory is None:
        if settings is None:
            raise RuntimeError(
                "get_session_factory() called without settings and no engine is "
                "cached. Pass the settings explicitly, or request the `engine` or "
                "`session` fixture so a test-scoped engine is registered first. "
                "Refusing here is what stops tests writing to the configured database."
            )
        _session_factory = sessionmaker(bind=get_engine(settings), expire_on_commit=False,
                                        future=True)
    return _session_factory


def reset_engine() -> None:
    """Drop the cached engine and session factory. Used by tests."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def create_all(engine: Engine) -> None:
    """Create tables directly.

    Alembic owns schema evolution; this exists for tests and first-run local
    setup, where running a migration is more ceremony than value. Once a
    migration exists for a table, prefer `alembic upgrade head`.
    """
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(settings: Settings | None = None) -> Iterator[Session]:
    """Session context manager that commits on success and rolls back on error.

    Defaults to the configured settings rather than to `None`. This is the CLI
    entry point -- every script in `scripts/` and `eval/` opens its session this way
    and genuinely wants whatever `DATABASE_URL` says, so resolving it here is what
    those callers mean.

    It is still not a silent fallback: `get_session_factory` only consults `settings`
    when no engine is cached yet, so under a test that has already registered a
    temp engine the argument is ignored and the test database wins. That is the
    distinction the guard exists to protect -- an *explicit* request for the
    configured database is fine, an implicit one from a test is not.
    """
    session = get_session_factory(settings if settings is not None else get_settings())()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Lives here rather than in any one API module because every router needs it and
    the alternative was importing it *from* a sibling router, which makes the set of
    API modules a cycle waiting to happen.

    The transaction is committed explicitly by the handlers rather than on context
    exit, so a per-item failure inside a batch (a rejected upload, one bad
    conversation id) can be rolled back and committed independently without
    unwinding the whole request.

    Uses the process-wide factory with no explicit settings, which is only safe
    because the application registers that factory with its own settings during
    startup. Passing settings here instead would make this dependency disagree
    with the engine the app actually uses.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
