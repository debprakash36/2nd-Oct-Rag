"""Alembic environment.

Two things here are load-bearing rather than boilerplate:

1. The database URL comes from `Settings`, not from `alembic.ini`. The app and the
   migrations must agree on which database they are pointed at, and a single
   source for that is the only way to guarantee it.

2. `render_as_batch` is enabled for SQLite. Alembic cannot ALTER a column on
   SQLite, so without batch mode any migration run against SQLite fails. The
   target is Postgres, but the test suite and local runs use SQLite, and a schema
   change that only works on the target database is untested until production.
"""

from __future__ import annotations

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.config import get_settings
from app.db.models import Base

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    return config.get_main_option("sqlalchemy.url") or get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL without connecting. Used by `alembic upgrade head --sql`."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection."""
    settings = get_settings()
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=True,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    # DDL is committed as it goes; releasing before exit avoids holding a
    # connection open on Windows file locks.
    connectable.dispose()

    if settings.environment in {"staging", "production"}:
        # pgvector is required on the target database. Checked here rather than
        # at model import so the failure names migrations, not the app.
        _assert_pgvector(connection)


def _assert_pgvector(connection) -> None:
    """Fail loudly if the target database lacks pgvector.

    Silently running without the extension would leave the `vector` column
    unavailable, and the failure would surface later as an indexing error
    halfway through a corpus ingest rather than as a clear startup problem.
    """
    from sqlalchemy import text

    exists = connection.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    if not exists:
        raise RuntimeError(
            "pgvector is not installed in the target database. Run "
            "CREATE EXTENSION vector; before migrating. Without it the embedding "
            "column cannot be created as vector(N)."
        )


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()