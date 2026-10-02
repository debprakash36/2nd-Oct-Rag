"""Phase 6 activity 1: the threshold re-sweep and its delta record.

The tool's value is entirely in the delta, so the property under test is that the
delta **refuses to claim movement it cannot justify**. Both bugs found while building
it were of exactly that kind:

1. `_best_score`-style misreading in the sibling tool, and here: `dataset_size`
   reported the file's line count rather than the number of questions evaluated, so
   a 40-question run recorded itself as 200 and the comparability guard waved
   through a delta between incomparable samples, calling it a REGRESSION.
2. The first version compared recall across different sample sizes with no guard at
   all, and reported +0.059 as an IMPROVED.

Both produced confident, wrong, *directionally convenient* output. That is the
failure mode implementation.md 8.1 warns about, so it is what these tests pin.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load("retune_threshold", "retune_threshold.py")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset.jsonl"
    rows = [{"question": f"question number {i}?"} for i in range(10)]
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    return path


class TestDatasetAccounting:
    def test_reports_the_full_dataset_when_unlimited(self, dataset):
        total, fingerprint = rt.dataset_size(dataset)
        assert total == 10
        assert fingerprint != "missing"

    def test_limit_is_reflected_in_the_count(self, dataset):
        """The bug: a subset recorded itself as the full set.

        That made a 40-question run indistinguishable from a 200-question one, so the
        comparability guard could not fire and the delta was reported anyway.
        """
        total, _ = rt.dataset_size(dataset, limit=4)
        assert total == 4, "the count must be what was evaluated, not the file length"

    def test_limit_also_changes_the_fingerprint(self, dataset):
        """A subset and the full set must not hash alike.

        Same reason: the fingerprint is the second half of the comparability guard.
        """
        _, full = rt.dataset_size(dataset)
        _, subset = rt.dataset_size(dataset, limit=4)
        assert full != subset

    def test_the_same_slice_always_fingerprints_identically(self, dataset):
        _, a = rt.dataset_size(dataset, limit=4)
        _, b = rt.dataset_size(dataset, limit=4)
        assert a == b, "an identical rerun must compare as like-for-like"

    def test_reformatting_does_not_change_the_fingerprint(self, dataset):
        """Whitespace is not a dataset change; a reworded question is."""
        _, before = rt.dataset_size(dataset)
        dataset.write_text(
            dataset.read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8"
        )
        _, after = rt.dataset_size(dataset)
        assert before == after

    def test_a_missing_dataset_is_reported_not_fatal(self, tmp_path):
        assert rt.dataset_size(tmp_path / "nope.jsonl") == (0, "missing")

    def test_malformed_lines_are_skipped(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"question": "ok"}\nnot json\n{"question": "ok2"}\n', encoding="utf-8")
        total, _ = rt.dataset_size(path)
        assert total == 2


class TestPointSemantics:
    def test_joint_requires_both_targets(self):
        both = rt.Point(0.1, 0.9, 0.2, recall_ok=True, band_ok=True)
        recall_only = rt.Point(0.1, 0.9, 0.05, recall_ok=True, band_ok=False)
        band_only = rt.Point(0.1, 0.8, 0.2, recall_ok=False, band_ok=True)
        neither = rt.Point(0.1, 0.8, 0.05, recall_ok=False, band_ok=False)
        assert both.joint
        assert not recall_only.joint
        assert not band_only.joint
        assert not neither.joint


class TestSnapshotRecord:
    def _points(self):
        return [
            rt.Point(0.05, 0.89, 0.205, recall_ok=True, band_ok=True),
            rt.Point(0.20, 0.80, 0.355, recall_ok=False, band_ok=False),
        ]

    @pytest.fixture
    def corpus(self, tmp_path):
        """A real, readable database.

        Not a bare path: `record` refuses to fingerprint a corpus it cannot open,
        which is deliberate -- a snapshot claiming to describe an unreadable corpus
        is exactly the fabricated record the delta guards exist to prevent.
        """
        import sqlite3

        path = tmp_path / "corpus.db"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY)")
        connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY)")
        connection.executemany(
            "INSERT INTO chunks (id) VALUES (?)", [(i,) for i in range(12)]
        )
        connection.executemany(
            "INSERT INTO documents (id) VALUES (?)", [(i,) for i in range(3)]
        )
        connection.commit()
        connection.close()
        return path

    def test_records_sample_size_and_fingerprint(self, corpus, tmp_path, monkeypatch):
        """Without these the delta cannot tell a comparable run from an incomparable one."""
        monkeypatch.setattr(rt, "HISTORY_PATH", tmp_path / "h.jsonl")
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        snapshot = rt.record(self._points(), corpus, "note", 200, "fp-200")
        assert snapshot["questions"] == 200
        assert snapshot["dataset_fingerprint"] == "fp-200"
        assert snapshot["commit"] == "abc1234"

    def test_records_the_corpus_shape_actually_swept(self, corpus, tmp_path, monkeypatch):
        """Corpus size is what attributes a recall change to ingestion vs code."""
        monkeypatch.setattr(rt, "HISTORY_PATH", tmp_path / "h.jsonl")
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        snapshot = rt.record(self._points(), corpus, "", 200, "fp")
        assert snapshot["corpus"] == {"chunks": 12, "documents": 3}

    def test_recommends_the_highest_recall_joint_point(
        self, corpus, tmp_path, monkeypatch
    ):
        """A recommendation, not just a table, so movement in it is reportable."""
        monkeypatch.setattr(rt, "HISTORY_PATH", tmp_path / "h.jsonl")
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        better = rt.Point(0.10, 0.92, 0.22, recall_ok=True, band_ok=True)
        snapshot = rt.record([*self._points(), better], corpus, "", 200, "fp")
        assert snapshot["recommended"] == 0.10
        assert snapshot["joint_count"] == 2

    def test_no_joint_point_records_a_null_recommendation(
        self, corpus, tmp_path, monkeypatch
    ):
        """`None` is a real answer: it means no threshold satisfies both targets."""
        monkeypatch.setattr(rt, "HISTORY_PATH", tmp_path / "h.jsonl")
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        snapshot = rt.record(
            [rt.Point(0.5, 0.2, 0.9, recall_ok=False, band_ok=False)],
            corpus,
            "",
            200,
            "fp",
        )
        assert snapshot["recommended"] is None

    def test_history_appends_rather_than_replaces(
        self, corpus, tmp_path, monkeypatch
    ):
        """Monthly runs accumulate. A file that only held the latest run would make
        the trend impossible to read."""
        path = tmp_path / "h.jsonl"
        monkeypatch.setattr(rt, "HISTORY_PATH", path)
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        rt.record(self._points(), corpus, "one", 200, "fp")
        rt.record(self._points(), corpus, "two", 200, "fp")
        assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2

    def test_a_truncated_line_does_not_discard_the_history(
        self, corpus, tmp_path, monkeypatch
    ):
        """An interrupted run must not cost every earlier snapshot.

        JSONL rather than one JSON array specifically so a partial write damages one
        line rather than the file.
        """
        path = tmp_path / "h.jsonl"
        monkeypatch.setattr(rt, "HISTORY_PATH", path)
        monkeypatch.setattr(rt, "_git", lambda *a: "abc1234")
        rt.record(self._points(), corpus, "good", 200, "fp")
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"partial": ')
        history = rt.load_history()
        assert len(history) == 1
        assert history[0]["note"] == "good"

    def test_missing_history_loads_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rt, "HISTORY_PATH", tmp_path / "absent.jsonl")
        assert rt.load_history() == []


class TestDeltaGuards:
    """The guards, asserted on their effects rather than their print strings."""

    def _snap(self, **kw):
        base = {
            "recorded_at": "2026-01-01T00:00:00+00:00",
            "commit": "abc1234",
            "corpus": {"chunks": 297, "documents": 113},
            "questions": 200,
            "dataset_fingerprint": "fp",
            "points": [rt.Point(0.1, 0.89, 0.21, True, True).as_dict()],
            "recommended": 0.1,
            "joint_count": 1,
        }
        base.update(kw)
        return base

    def test_a_different_sample_size_is_not_comparable(self, capsys):
        before = self._snap(questions=200)
        after = self._snap(questions=40, dataset_fingerprint="other")
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "NOT LIKE-FOR-LIKE" in out
        assert "sample size changed" in out
        # The specific lie this prevents: a movement verdict on incomparable samples.
        assert "IMPROVED" not in out
        assert "REGRESSED" not in out

    def test_an_edited_dataset_is_not_comparable(self, capsys):
        before = self._snap()
        after = self._snap(dataset_fingerprint="edited")
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "NOT LIKE-FOR-LIKE" in out
        assert "the dataset changed" in out
        assert "IMPROVED" not in out
        assert "REGRESSED" not in out

    def test_a_small_movement_is_reported_as_noise(self, capsys):
        """200 questions means one question is worth 0.005 of recall@10."""
        before = self._snap()
        after = self._snap(
            points=[rt.Point(0.1, 0.895, 0.21, True, True).as_dict()]
        )
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "within noise" in out
        assert "IMPROVED" not in out

    def test_a_real_movement_is_reported(self, capsys):
        before = self._snap()
        after = self._snap(points=[rt.Point(0.1, 0.70, 0.21, False, True).as_dict()])
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "REGRESSED" in out

    def test_a_corpus_change_is_attributed_to_the_corpus(self, capsys):
        """The same numbers mean different things after the corpus changed."""
        before = self._snap()
        after = self._snap(
            corpus={"chunks": 400, "documents": 150},
            points=[rt.Point(0.1, 0.70, 0.21, False, False).as_dict()],
        )
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "REGRESSION" in out
        assert "corpus changed" in out
        assert "code regression" not in out

    def test_a_code_change_without_a_corpus_change_is_a_code_regression(self, capsys):
        before = self._snap()
        after = self._snap(
            commit="def9999",
            points=[rt.Point(0.1, 0.70, 0.21, False, False).as_dict()],
        )
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "code regression" in out

    def test_neither_changing_is_called_untraceable(self, capsys):
        """Numbers moved with nothing to attribute it to. That is a finding."""
        before = self._snap()
        after = self._snap(points=[rt.Point(0.1, 0.70, 0.21, False, False).as_dict()])
        rt.show_delta(before, after)
        out = capsys.readouterr().out
        assert "no traceable reason" in out

    def test_a_moved_recommendation_is_called_out(self, capsys):
        before = self._snap(recommended=0.1)
        after = self._snap(recommended=0.15)
        rt.show_delta(before, after)
        assert "RECOMMENDATION MOVED" in capsys.readouterr().out