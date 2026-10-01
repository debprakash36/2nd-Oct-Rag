"""RRF fusion and near-duplicate dedupe (architecture.md 3.3 stage 2)."""

from __future__ import annotations

import pytest

from app.retrieval.fusion import deduplicate, reciprocal_rank_fusion
from app.retrieval.types import RetrievalCandidate, Stage


def _cand(
    chunk_id: str, *, doc_id: str = "doc1", index: int = 0, text: str = "body"
) -> RetrievalCandidate:
    return RetrievalCandidate(chunk_id=chunk_id, doc_id=doc_id, chunk_index=index, text=text)


class TestReciprocalRankFusion:
    def test_uses_ranks_not_raw_scores(self):
        """RRF's whole point (architecture.md 3.3): incomparable scales cancel out.

        The vector list's scores are on a 0-1 cosine scale and the keyword list's
        are raw BM25 magnitudes, but the fused order depends only on rank position.
        If fusion regressed to score combination, this test would fail.
        """
        vector_first = _cand("a")
        vector_first.scores[Stage.VECTOR] = 0.9
        vector_second = _cand("b")
        vector_second.scores[Stage.VECTOR] = 0.8

        keyword_first = _cand("c")
        keyword_first.scores[Stage.KEYWORD] = 100.0
        keyword_second = _cand("d")
        keyword_second.scores[Stage.KEYWORD] = 5.0

        fused = reciprocal_rank_fusion(
            {
                Stage.VECTOR: [vector_first, vector_second],
                Stage.KEYWORD: [keyword_first, keyword_second],
            }
        )

        # `a` and `c` are both rank 1 in exactly one list, so they tie despite a
        # BM25 of 100.0 against a cosine of 0.9. That equality is the property
        # being asserted: fusion consumed ranks and discarded the raw magnitudes.
        assert len(fused) == 4
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["a"].scores[Stage.FUSED] == pytest.approx(
            by_id["c"].scores[Stage.FUSED]
        )
        assert by_id["b"].scores[Stage.FUSED] == pytest.approx(
            by_id["d"].scores[Stage.FUSED]
        )
        # Rank 1 beats rank 2, and the top is not the high-BM25 candidate on its own.
        assert by_id["a"].scores[Stage.FUSED] > by_id["b"].scores[Stage.FUSED]

    def test_candidate_in_both_lists_outranks_single_list_candidate(self):
        shared = _cand("shared")
        only_vector = _cand("v")
        only_keyword = _cand("k")

        fused = reciprocal_rank_fusion(
            {
                Stage.VECTOR: [shared, only_vector],
                Stage.KEYWORD: [shared, only_keyword],
            }
        )

        assert fused[0].chunk_id == "shared"
        assert shared.ranks[Stage.VECTOR] == 1
        assert shared.ranks[Stage.KEYWORD] == 1

    def test_single_list_candidate_is_not_penalised_against_zero(self):
        """A chunk found by one stage only must still be scoreable.

        Treating a missing list as a zero score would push every keyword-only hit
        below every vector hit, which defeats the point of hybrid search.
        """
        both = _cand("both", index=0)
        single = _cand("single", index=1)

        fused = reciprocal_rank_fusion(
            {Stage.VECTOR: [both, single], Stage.KEYWORD: [both]}
        )
        by_id = {c.chunk_id: c for c in fused}
        assert by_id["both"].scores[Stage.FUSED] > by_id["single"].scores[Stage.FUSED]

    def test_empty_lists_produce_no_candidates(self):
        assert reciprocal_rank_fusion({Stage.VECTOR: [], Stage.KEYWORD: []}) == []

    def test_weights_shift_order(self):
        a = _cand("a", index=0)
        b = _cand("b", index=1)
        # `a` ranks first in vector, `b` first in keyword. Weighting keyword
        # heavily must flip the order.
        fused = reciprocal_rank_fusion(
            {Stage.VECTOR: [a, b], Stage.KEYWORD: [b, a]},
            weights={Stage.VECTOR: 0.1, Stage.KEYWORD: 10.0},
        )
        assert fused[0].chunk_id == "b"


class TestDeduplicate:
    def test_removes_near_identical_text(self):
        """Boilerplate repeated across documents must not fill the context."""
        first = _cand("a", text="This document is governed by the master terms.")
        second = _cand("b", doc_id="doc2", text="this document is governed by the MASTER terms")

        out = deduplicate([first, second])
        assert [c.chunk_id for c in out] == ["a"]

    def test_keeps_genuinely_distinct_text(self):
        out = deduplicate(
            [
                _cand("a", text="Refunds are available within 30 days."),
                _cand("b", text="Shipping takes three to five business days."),
            ]
        )
        assert len(out) == 2

    def test_keeps_highest_ranked_copy(self):
        """Input is relevance-ordered, so the first occurrence is the best one."""
        dup_a = _cand("a", text="identical boilerplate block of text")
        dup_b = _cand("b", doc_id="doc2", text="identical boilerplate block of text")
        out = deduplicate([dup_a, dup_b])
        assert out[0].chunk_id == "a"

    def test_short_chunks_not_treated_as_duplicates(self):
        """Coincidental collisions on tiny text must not delete real content."""
        out = deduplicate([_cand("a", text="See 4."), _cand("b", doc_id="d2", text="See 5.")])
        assert len(out) == 2

    def test_limit_is_respected(self):
        cands = [
            _cand(f"c{i}", doc_id=f"d{i}", index=i, text=f"unique passage {i}")
            for i in range(10)
        ]
        assert len(deduplicate(cands, limit=3)) == 3


