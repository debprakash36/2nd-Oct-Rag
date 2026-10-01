"""Phase 6 tooling: the pilot gate and the improvement-loop classifier (8.1, 8.2).

The property under test is not arithmetic. It is **refusal to report a number that
would clear a gate it has not earned.** A pilot gate that reports "100% thumbs up"
from four votes on a single query is worse than one that reports nothing, because it
converts an absence of evidence into a pass.

These tests build synthetic `query_logs` and assert both directions: real samples
produce PASS/FAIL, thin samples produce UNMEASURED even when the rate looks perfect.
"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    """Import a `scripts/` module by path.

    `scripts/` is not a package, so a normal import would not resolve. Loading by
    path keeps the tooling importable from tests without adding `scripts/__init__.py`
    and changing how every other script is run.
    """
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pilot = _load("pilot_metrics", "pilot_metrics.py")
gaps = _load("content_gaps", "content_gaps.py")


@pytest.fixture
def log_db(tmp_path: Path):
    """An empty query_logs table with the real schema."""
    db = tmp_path / "pilot.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE query_logs ("
        "  query_id TEXT, original_query TEXT, abstained INTEGER DEFAULT 0,"
        "  feedback TEXT, ttft_ms INTEGER, total_ms INTEGER, created_at TEXT)"
    )
    connection.commit()
    connection.close()
    return db


def _add(db: Path, query: str, *, abstained=0, feedback=None, ttft=None) -> None:
    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT INTO query_logs (query_id, original_query, abstained, feedback, ttft_ms)"
        " VALUES (?, ?, ?, ?, ?)",
        (f"q{abs(hash(query)) % 10**8}", query, abstained, feedback, ttft),
    )
    connection.commit()
    connection.close()


def _metrics(db: Path, min_samples: int = 20):
    return pilot.build_metrics(pilot.collect(db), min_samples)


def _state(metrics, name: str) -> str:
    return next(m.state for m in metrics if m.name == name)


class TestPilotGateRefusesUnearnedPasses:
    def test_a_perfect_rate_on_one_query_is_unmeasured(self, log_db):
        """The exact trap: 100% thumbs-up, from a single interaction.

        This is what the real checkout looks like, and reporting it as a pass would
        clear the 70% gate on the strength of one test.
        """
        _add(log_db, "what is the refund window", feedback="up", ttft=900)
        for _ in range(3):
            _add(log_db, "what is the refund window", feedback="up", ttft=900)

        metrics = _metrics(log_db)
        helpful = next(m for m in metrics if m.name == "answered-helpfully")
        assert helpful.state == "unmeasured"
        assert helpful.value == 1.0, "the raw number exists; it is just not reportable"
        assert "Not reportable" in helpful.detail

    def test_enough_distinct_queries_passes(self, log_db):
        for i in range(30):
            _add(log_db, f"question {i}", feedback="up", ttft=800)
        assert _state(_metrics(log_db), "answered-helpfully") == "pass"

    def test_enough_queries_with_mostly_down_fails(self, log_db):
        for i in range(10):
            _add(log_db, f"up {i}", feedback="up", ttft=800)
        for i in range(30):
            _add(log_db, f"down {i}", feedback="down", ttft=800)

        metrics = _metrics(log_db)
        helpful = next(m for m in metrics if m.name == "answered-helpfully")
        assert helpful.state == "fail"
        assert helpful.value == 0.25

    def test_no_votes_is_unmeasured_not_a_pass(self, log_db):
        for i in range(30):
            _add(log_db, f"question {i}", abstained=0, ttft=800)
        assert _state(_metrics(log_db), "answered-helpfully") == "unmeasured"

    def test_repeated_votes_do_not_inflate_the_sample(self, log_db):
        """One heavily-interacted query is still one query.

        Voting 50 times on a single question is not 50 samples of satisfaction.
        """
        for _ in range(50):
            _add(log_db, "one question", feedback="up", ttft=800)
        helpful = next(m for m in _metrics(log_db) if m.name == "answered-helpfully")
        assert helpful.state == "unmeasured"
        assert helpful.samples == 1


class TestPilotGateBands:
    def test_refusal_rate_inside_the_band_passes(self, log_db):
        for i in range(80):
            _add(log_db, f"q{i}", abstained=1 if i % 5 == 0 else 0, ttft=700)
        assert _state(_metrics(log_db), "refusal rate") == "pass"

    def test_a_near_zero_refusal_rate_fails_and_says_why(self, log_db):
        """PRD 8.2: a near-zero refusal rate is the most damaging failure mode.

        It must not be reported as good news, because it usually means the threshold
        is too permissive and the system is answering from irrelevant text.
        """
        for i in range(40):
            _add(log_db, f"q{i}", abstained=0, ttft=700)
        metrics = _metrics(log_db)
        refusal = next(m for m in metrics if m.name == "refusal rate")
        assert refusal.state == "fail"
        assert "too permissive" in refusal.detail

    def test_an_almost_total_refusal_rate_fails(self, log_db):
        for i in range(40):
            _add(log_db, f"q{i}", abstained=1, ttft=700)
        assert _state(_metrics(log_db), "refusal rate") == "fail"

    def test_ttft_p95_uses_the_same_percentile_as_the_eval_harness(self, log_db):
        for i in range(100):
            _add(log_db, f"q{i}", ttft=1000 + i * 10)
        metrics = _metrics(log_db)
        ttft = next(m for m in metrics if m.name == "TTFT p95")
        # 100 samples, 0..990 step 10 -> p95 sits at position 0.95*99 = 94.05.
        assert ttft.state == "pass"
        assert ttft.value == pytest.approx(1940.5, abs=1.0)

    def test_a_slow_p95_fails(self, log_db):
        for i in range(30):
            _add(log_db, f"q{i}", ttft=6000)
        assert _state(_metrics(log_db), "TTFT p95") == "fail"


class TestPilotGateDegradesCleanly:
    def test_a_missing_table_is_reported_not_crashed(self, tmp_path):
        empty = tmp_path / "empty.db"
        sqlite3.connect(empty).close()
        metrics = _metrics(empty)
        assert len(metrics) == 1
        assert metrics[0].state == "unmeasured"
        assert "could not read query_logs" in metrics[0].detail


class TestGapClassifierTaxonomy:
    """The classifier's value is entirely in not confusing the categories.

    implementation.md 8.1: a content gap and a retrieval gap need different fixes, and
    "an agent that improves retrieval for a content gap will change nothing and
    report progress."
    """

    def test_answered_queries_are_an_answer_gap_not_a_retrieval_one(self):
        """If the gate passed, retrieval worked. Whatever else is wrong, tuning the
        threshold cannot help."""
        class Result:
            abstained = False
            candidates: ClassVar[list] = []

        class Candidate:
            def best_score(self):
                return 0.4

        Result.candidates = [Candidate(), Candidate()]
        assert gaps._best_score(Result()) == pytest.approx(0.4)
        assert gaps.NOISE_FLOOR < 0.4

    def test_best_score_reads_candidates_not_the_result(self):
        """Regression: `_best_score` originally read `result.scores`, which does not
        exist on `RetrievalResult` -- it is a per-candidate accumulator. Every query
        therefore scored 0.0 and every diagnosis came back "content gap", which is
        the most dangerous possible error for this tool."""
        class Candidate:
            def best_score(self):
                return 0.29

        class Result:
            scores: ClassVar[dict] = {"vector": 0.29}  # wrong place; must be ignored
            candidates: ClassVar[list] = [Candidate()]

        assert gaps._best_score(Result()) == pytest.approx(0.29)

    def test_best_score_of_no_candidates_is_zero(self):
        class Result:
            candidates: ClassVar[list] = []

        assert gaps._best_score(Result()) == 0.0

    def test_a_candidate_without_best_score_is_skipped_not_fatal(self):
        class Result:
            candidates: ClassVar[list] = [object(), object()]

        assert gaps._best_score(Result()) == 0.0

    def test_the_taxonomy_documents_every_classification_it_can_emit(self):
        """The table in the module docstring is the contract an operator reads.

        If a classification is added without a docstring row, the fix guidance for
        it is invisible to the person who needs it -- which is the whole failure
        mode 8.1 warns about.
        """
        emitted = {
            "answer_gap",
            "content_gap",
            "threshold_gap",
            "lexical_gap",
            "semantic_gap",
            "ranking_gap",
        }
        for classification in emitted:
            assert f"`{classification}`" in gaps.__doc__, (
                f"{classification} is emitted but has no row in the docstring table"
            )

    def test_noise_floor_and_strong_signal_bracket_the_realistic_range(self):
        assert gaps.NOISE_FLOOR < gaps.STRONG_SIGNAL
        assert gaps.STRONG_SIGNAL <= 1.0


class TestGapReviewSources:
    def test_reads_refusals_and_down_votes_from_the_log(self, log_db, monkeypatch):
        _add(log_db, "refused question", abstained=1)
        _add(log_db, "disliked question", feedback="down")
        _add(log_db, "happy question", feedback="up")

        monkeypatch.setattr(gaps, "REPO_ROOT", log_db.parent)
        (log_db.parent / "rag.db").write_bytes(log_db.read_bytes())
        found = gaps.queries_from_log(10)
        queries = {q for q, _ in found}
        assert "refused question" in queries
        assert "disliked question" in queries
        assert "happy question" not in queries, "a liked query is not a review target"

    def test_deduplicates_repeated_questions(self, log_db, monkeypatch):
        for _ in range(5):
            _add(log_db, "same problem", abstained=1)
        monkeypatch.setattr(gaps, "REPO_ROOT", log_db.parent)
        (log_db.parent / "rag.db").write_bytes(log_db.read_bytes())
        assert len(gaps.queries_from_log(10)) == 1

    def test_a_missing_log_yields_no_targets_rather_than_raising(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gaps, "REPO_ROOT", tmp_path)
        assert gaps.queries_from_log(10) == []
