"""The test suite must not write to the configured database.

`tests/conftest.py` claims "Every test runs against a fresh temp SQLite file with a
pinned embedding dimension." That claim was false, and the cost was not visible: the
suite stayed green while leaving 64-wide embeddings in the real `rag.db`, which
production reads at 384. One wrong-width row raises inside the parallel search stage
and aborts every retrieval query, so a green test suite coexisted with a completely
dead search path.

The mechanism was `get_session_factory()` called with no argument. It falls through
to `get_settings()`, which is the module-level singleton -- *not* the object the
`client` fixture overrides via `dependency_overrides`. So the result depended on
whether some earlier test had already warmed the global engine cache, and
`settings_env` teardown clears that cache between tests. `tests/api/test_conversations.py`
had two such calls. It survived because the write is small, order-dependent, and
`rag.db` is a working-tree artifact nobody diffs.

These tests pin the invariant shut.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_session_factory, reset_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGURED_DB = REPO_ROOT / "rag.db"


class TestSessionFactoryRefusesAnImplicitDatabase:
    def test_raises_when_called_without_settings_and_no_engine_is_cached(self):
        """The guard itself: no settings and a cold cache is a bug, not a default.

        Without this the call silently resolves to whatever `DATABASE_URL` says. In
        the test environment that is the developer's real database.
        """
        reset_engine()
        with pytest.raises(RuntimeError, match="without settings"):
            get_session_factory()
        reset_engine()

    def test_explicit_settings_still_work(self, settings_env: Settings):
        reset_engine()
        session = get_session_factory(settings_env)()
        try:
            # Compared as resolved paths, not as URL strings: SQLAlchemy
            # percent-encodes a Windows drive letter in the rendered URL, so a string
            # comparison fails on a path that is in fact the right database.
            bound = Path(str(session.get_bind().url.database))
            expected = Path(settings_env.database_url.split(":///")[-1])
            assert bound.resolve() == expected.resolve()
        finally:
            session.close()
            reset_engine()

    def test_a_warmed_factory_is_reused(self, engine, settings_env: Settings):
        """Once an engine is registered, the no-argument call is legitimate.

        Several tests reach for the global factory for convenience. That is fine --
        what must never happen is the *first* call in a process resolving against the
        configured database.
        """
        get_session_factory(settings_env)  # what the `engine` fixture now does
        assert get_session_factory() is not None


class TestTestDatabaseIsolation:
    def test_session_scope_resolves_the_configured_database(self):
        """A CLI caller opening `session_scope()` wants `DATABASE_URL`, not a failure.

        Regression: the guard in `get_session_factory` initially broke every script in
        `scripts/` and `eval/`, all of which open their session with a bare
        `session_scope()`. The guard was right about the test path and wrong about
        this one -- a script asking for the configured database is asking on purpose.

        The distinction is the warm cache: settings are only consulted when no engine
        is registered, so a test that has already registered a temp engine still gets
        it. That is what the next test pins.
        """
        from app.db.session import session_scope

        reset_engine()
        try:
            with session_scope() as probe:
                bound = Path(str(probe.get_bind().url.database)).resolve()
            expected = Path(get_settings().database_url.split(":///")[-1]).resolve()
            assert bound == expected
        finally:
            reset_engine()

    def test_a_warmed_test_engine_still_wins_over_the_configured_database(
        self, settings_env: Settings
    ):
        """The other half, and the reason the guard is safe to keep.

        Under a test that has already registered a temp engine, a bare
        `session_scope()` must use it -- otherwise the isolation fix would have moved
        the leak rather than closed it.
        """
        from app.db.session import get_session_factory, session_scope

        get_session_factory(settings_env)  # warm the cache, as `engine` does
        with session_scope() as probe:
            bound = Path(str(probe.get_bind().url.database)).resolve()
        expected = Path(settings_env.database_url.split(":///")[-1]).resolve()
        assert bound == expected, "a test session must not reach the configured database"

    def test_the_engine_fixture_registers_the_session_factory(self, engine, session: Session):
        """The fixture that establishes the test database must establish the factory.

        Otherwise a test can hold a test-scoped engine and still write through a
        factory bound somewhere else -- the exact split that let this happen.
        """
        factory = get_session_factory()
        probe = factory()
        try:
            assert probe.get_bind() is engine or str(probe.get_bind().url) == str(engine.url)
        finally:
            probe.close()

    def test_configured_database_is_untouched_by_an_api_run(self, tmp_path: Path):
        """Run a representative slice of `tests/api` and require no change to rag.db.

        A subprocess is required: the leak depended on global engine state, which
        does not survive into this test's already-warm process. The child inherits a
        clean slate, which is exactly the condition that triggered the original bug.
        """
        import subprocess
        import sys

        if not CONFIGURED_DB.exists():
            pytest.skip("no rag.db in the working tree; nothing to protect")

        before = CONFIGURED_DB.read_bytes()
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/api/test_conversations.py",
                "-q",
                "--no-header",
                "-p",
                "no:randomly",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        after = CONFIGURED_DB.read_bytes()

        assert result.returncode == 0, (
            f"the conversation tests did not pass, so this check proves nothing:\n"
            f"{result.stdout[-2000:]}"
        )
        assert before == after, (
            "running tests/api/test_conversations.py modified rag.db. A test is "
            "writing to the configured database instead of its temp file -- almost "
            "certainly a get_session_factory() call with no settings argument made "
            "before the test engine is registered."
        )

    @pytest.mark.indexcheck
    def test_configured_database_has_no_short_embeddings(self):
        """The corruption the leak produced. Opt-in: it reports working-tree state.

        This is a reminder that the corpus needs a deliberate repair, not a code
        invariant, so it is marked `indexcheck` and excluded from the default suite.
        A test that fails on pre-existing data damage in a file nobody diffs gets
        deleted rather than fixed, and the damage would outlive it.

        The companion test above -- that a test run leaves `rag.db` byte-identical --
        is the one that must run every time, because that is the regression this whole
        file exists to prevent.
        """
        if not CONFIGURED_DB.exists():
            pytest.skip("no rag.db in the working tree")

        connection = sqlite3.connect(CONFIGURED_DB)
        try:
            rows = connection.execute(
                "SELECT c.chunk_id, length(c.embedding) FROM chunks c "
                "WHERE c.embedding IS NOT NULL AND length(c.embedding) < 3000"
            ).fetchall()
        finally:
            connection.close()

        assert not rows, (
            f"{len(rows)} chunk(s) in rag.db hold a short embedding "
            f"(e.g. {rows[0][0]}, {rows[0][1]} chars). A 384-wide query against these "
            f"raises and aborts every retrieval query. These came from tests writing "
            f"to the configured database with the test suite's pinned "
            f"embedding_dim=64. Inspect with: "
            f"python scripts/purge_bad_dimension_chunks.py"
        )
