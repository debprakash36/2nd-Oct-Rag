"""The re-embed script's decisions, tested without touching the network.

`scripts/reembed_chunks.py` rewrites every vector in a corpus, so the two places
it can be subtly wrong are both about *which rows it decides to touch*:

1. The resume filter. This is not hypothetical. The first version of that filter
   read `embedding_model IS ?` when it needed `!= ?`, so it selected exactly the
   rows already at the target model and skipped the entire `fake-embed-v1`
   remainder. On this corpus it reported "2 already at target; 13 to convert" and
   a naive reading is that only 13 rows were ever wrong. The width disjunct could
   not rescue it, because all 295 wrong-model rows are already 384-wide --
   correct width, meaningless vectors. These tests pin the filter directly.

2. Atomicity of a batch write. The vector and its `embedding_model` label go in
   one statement, because a row claiming to be at the target model while holding
   something else would be skipped by every future resume, permanently.

No test here calls the hosted API. `embed_batch` is exercised through a stub
provider so the suite stays offline and deterministic.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "reembed_chunks.py"


def _load_module() -> Any:
    """Import the script by path.

    Loaded rather than imported normally because `scripts/` is not a package, and
    the script's own imports of `app` are deliberately deferred into the functions
    that need them -- so importing it must not drag in settings or a network
    client.
    """
    spec = importlib.util.spec_from_file_location("reembed_chunks", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["reembed_chunks"] = module
    spec.loader.exec_module(module)
    return module


reembed = _load_module()

FAKE_MODEL = "fake-embed-v1"
TARGET_MODEL = reembed.TARGET_MODEL
TARGET_DIM = reembed.TARGET_DIM


def _make_db(tmp_path: Path, rows: list[tuple[str, str, str]]) -> sqlite3.Connection:
    """A minimal `chunks` table matching the real schema's relevant columns."""
    db_path = tmp_path / "test.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE chunks ("
        "  chunk_id TEXT PRIMARY KEY,"
        "  doc_id TEXT,"
        "  chunk_index INTEGER,"
        "  text TEXT,"
        "  token_count INTEGER,"
        "  embedding TEXT,"
        "  embedding_model TEXT"
        ")"
    )
    connection.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    connection.commit()
    return connection


def _vec(width: int) -> str:
    import json

    return json.dumps([0.1] * width)


def _row(
    chunk_id: str, model: str | None, width: int
) -> tuple[str, str, str, str, int, str, str | None]:
    return (chunk_id, "d1", 0, f"text of {chunk_id}", 5, _vec(width), model)


class TestPendingSelection:
    """Which rows the script decides to convert."""

    def test_convert_everything_without_resume(self, tmp_path):
        connection = _make_db(
            tmp_path,
            [
                _row("c1", FAKE_MODEL, TARGET_DIM),
                _row("c2", FAKE_MODEL, 64),
                _row("c3", TARGET_MODEL, TARGET_DIM),
            ],
        )
        rows = reembed.pending(connection, resume=False)
        assert {r.chunk_id for r in rows} == {"c1", "c2", "c3"}

    def test_resume_skips_only_rows_done_in_both_respects(self, tmp_path):
        """The regression that motivated this file.

        A row is done only when the model *and* the width both match. The 295
        wrong-model/right-width rows are the trap: they are already 384-dim, so a
        filter that only checks width reports the corpus as converted while every
        vector in it is still a hash.
        """
        connection = _make_db(
            tmp_path,
            [
                _row("done", TARGET_MODEL, TARGET_DIM),
                _row("wrong_model_right_width", FAKE_MODEL, TARGET_DIM),
                _row("right_model_wrong_width", TARGET_MODEL, 64),
                _row("wrong_both", FAKE_MODEL, 64),
            ],
        )
        rows = reembed.pending(connection, resume=True)
        assert {r.chunk_id for r in rows} == {
            "wrong_model_right_width",
            "right_model_wrong_width",
            "wrong_both",
        }

    def test_resume_includes_rows_with_a_null_model(self, tmp_path):
        """`!=` against NULL yields NULL, not TRUE.

        A NULL-labelled row has to be selected explicitly or a corpus with any
        unlabelled rows would resume as "nothing to do".
        """
        connection = _make_db(
            tmp_path,
            [_row("c1", None, TARGET_DIM)],
        )
        assert {r.chunk_id for r in reembed.pending(connection, resume=True)} == {"c1"}

    def test_resume_is_empty_when_everything_is_converted(self, tmp_path):
        connection = _make_db(
            tmp_path,
            [_row(f"c{i}", TARGET_MODEL, TARGET_DIM) for i in range(5)],
        )
        assert reembed.pending(connection, resume=True) == []


class TestVerify:
    """A partial conversion must never pass as success."""

    def test_reports_ok_only_when_fully_converted(self, tmp_path):
        connection = _make_db(
            tmp_path,
            [_row(f"c{i}", TARGET_MODEL, TARGET_DIM) for i in range(3)],
        )
        ok, detail = reembed.verify(connection)
        assert ok
        assert "3/3" in detail

    def test_partial_conversion_fails(self, tmp_path):
        connection = _make_db(
            tmp_path,
            [
                _row("c1", TARGET_MODEL, TARGET_DIM),
                _row("c2", FAKE_MODEL, TARGET_DIM),
            ],
        )
        ok, detail = reembed.verify(connection)
        assert not ok
        assert "1/2" in detail
        # The detail has to name the leftover, or an operator cannot tell which
        # rows to look at.
        assert FAKE_MODEL in detail

    def test_short_vector_fails_even_when_the_label_says_converted(self, tmp_path):
        """The width half of the check, independent of the label."""
        connection = _make_db(tmp_path, [_row("c1", TARGET_MODEL, 64)])
        ok, _ = reembed.verify(connection)
        assert not ok


class TestSurvey:
    def test_groups_by_model_and_width(self, tmp_path):
        connection = _make_db(
            tmp_path,
            [
                _row("c1", FAKE_MODEL, TARGET_DIM),
                _row("c2", FAKE_MODEL, TARGET_DIM),
                _row("c3", FAKE_MODEL, 64),
            ],
        )
        assert reembed.survey(connection) == {
            f"{FAKE_MODEL} @ {TARGET_DIM}": 2,
            f"{FAKE_MODEL} @ 64": 1,
        }


class TestWriteBatch:
    def test_vector_and_model_land_together(self, tmp_path):
        connection = _make_db(tmp_path, [_row("c1", FAKE_MODEL, 64)])
        rows = reembed.pending(connection, resume=False)
        vector = [0.5] * TARGET_DIM
        reembed.write_batch(connection, rows, [vector])

        stored_model, stored = connection.execute(
            "SELECT embedding_model, embedding FROM chunks WHERE chunk_id='c1'"
        ).fetchone()
        import json

        assert stored_model == TARGET_MODEL
        assert json.loads(stored) == vector

    def test_rolled_back_batch_leaves_nothing_written(self, tmp_path):
        """A failed batch must not half-apply.

        Simulated with a subclass whose `executemany` writes the first row for
        real and then raises, which is the shape of the failure being guarded
        against: the first rows of a batch already on disk when the commit never
        happens. Monkeypatching the connection is not an option --
        `executemany` is read-only on `sqlite3.Connection` -- so the subclass
        intercepts the call instead.
        """

        class ExplodingConnection(sqlite3.Connection):
            rows_written = 0

            def executemany(self, sql, seq):
                seq = list(seq)
                super().executemany(sql, seq[:1])
                ExplodingConnection.rows_written = 1
                raise sqlite3.IntegrityError("simulated failure mid-batch")

        db_path = tmp_path / "test.db"
        connection = sqlite3.connect(db_path, factory=ExplodingConnection)
        connection.execute(
            "CREATE TABLE chunks ("
            "  chunk_id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER,"
            "  text TEXT, token_count INTEGER, embedding TEXT, embedding_model TEXT)"
        )
        connection.execute(
            "INSERT INTO chunks "
            "VALUES ('c1','d1',0,'text',5,?,?)",
            (_vec(64), FAKE_MODEL),
        )
        connection.commit()

        rows = reembed.pending(connection, resume=False)
        with pytest.raises(sqlite3.IntegrityError):
            reembed.write_batch(connection, rows, [[0.5] * TARGET_DIM] * len(rows))

        assert ExplodingConnection.rows_written == 1, "the stub did not fail midway"
        model_after, = connection.execute(
            "SELECT embedding_model FROM chunks WHERE chunk_id='c1'"
        ).fetchone()
        assert model_after == FAKE_MODEL, "partial write was not rolled back"


class TestEmbedBatch:
    """Retry classification, without a network."""

    class _Stub:
        def __init__(self, responses: list[Any]) -> None:
            self.responses = responses
            self.calls = 0

        def embed(self, texts, *, model):
            self.calls += 1
            outcome = self.responses[min(self.calls - 1, len(self.responses) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def _embedding_error(self, message: str) -> Exception:
        from app.core.errors import EmbeddingError

        return EmbeddingError(message)

    def test_returns_vectors_on_first_success(self):
        vectors = [[0.1] * TARGET_DIM]
        stats = reembed.RunStats()
        stub = self._Stub([vectors])
        got = reembed.embed_batch(stub, ["x"], retries=2, stats=stats)
        assert got == vectors
        assert stub.calls == 1
        assert not stats.failures

    def test_retries_a_rate_limit_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(reembed, "BASE_BACKOFF", 0.001)
        responses = [
            self._embedding_error("huggingface returned HTTP 429"),
            [[0.1] * TARGET_DIM],
        ]
        stats = reembed.RunStats()
        stub = self._Stub(responses)
        got = reembed.embed_batch(stub, ["x"], retries=3, stats=stats)
        assert got == [[0.1] * TARGET_DIM]
        assert stub.calls == 2
        assert stats.retries == 1

    def test_does_not_retry_an_auth_failure(self, monkeypatch):
        """401 is the token; retrying it only delays the real diagnosis."""
        monkeypatch.setattr(reembed, "BASE_BACKOFF", 0.001)
        stats = reembed.RunStats()
        stub = self._Stub([self._embedding_error("huggingface returned HTTP 401")])
        got = reembed.embed_batch(stub, ["x"], retries=5, stats=stats)
        assert got is None
        assert stub.calls == 1
        assert stats.retries == 0

    def test_gives_up_after_exhausting_retries(self, monkeypatch):
        monkeypatch.setattr(reembed, "BASE_BACKOFF", 0.001)
        stats = reembed.RunStats()
        stub = self._Stub([self._embedding_error("huggingface returned HTTP 503")])
        got = reembed.embed_batch(stub, ["x"], retries=2, stats=stats)
        assert got is None
        assert stub.calls == 3, "initial attempt plus two retries"

    def test_rejects_a_short_vector(self):
        """A 200 with the wrong width is still a failure.

        The hosted model is pinned at 384, so this would mean the response was
        not what was asked for -- writing it would produce a corpus that passes
        the width check on paper and fails at query time.
        """
        stats = reembed.RunStats()
        stub = self._Stub([[[0.1] * 64]])
        got = reembed.embed_batch(stub, ["x"], retries=0, stats=stats)
        assert got is None
        assert stats.failures

    def test_rejects_a_count_mismatch(self):
        stats = reembed.RunStats()
        stub = self._Stub([[[0.1] * TARGET_DIM]])
        got = reembed.embed_batch(stub, ["a", "b"], retries=0, stats=stats)
        assert got is None
        assert stats.failures