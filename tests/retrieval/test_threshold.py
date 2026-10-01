"""Relevance threshold and context budget (FR-14, FR-15).

The distinction under test throughout: the threshold is a *quality gate* that can
abstain, and the budget is a *capacity constraint* that only evicts. A run that
evicts and answers is a different outcome from one that abstains, and conflating
them would make an eval threshold sweep look like it achieved two things at once.
"""

from __future__ import annotations

import pytest

from app.retrieval.threshold import (
    MAX_CHUNK_TOKENS,
    apply_threshold,
    apply_token_budget,
    estimate_tokens,
    score_for_threshold,
    stage_of_score,
)
from app.retrieval.types import RetrievalCandidate, Stage


def _cand(
    chunk_id: str,
    score: float,
    *,
    tokens: int = 100,
    text: str = "body",
    doc_id: str = "d1",
    index: int = 0,
) -> RetrievalCandidate:
    c = RetrievalCandidate(
        chunk_id=chunk_id, doc_id=doc_id, chunk_index=index, text=text, token_count=tokens
    )
    c.scores[Stage.RERANKED] = score
    return c


class TestApplyThreshold:
    def test_keeps_candidates_above_threshold(self):
        kept, abstained, _ = apply_threshold([_cand("a", 0.9)], threshold=0.5)
        assert not abstained
        assert len(kept) == 1

    def test_abstains_below_threshold(self):
        """FR-14: no LLM call, no tokens, no chance of a plausible invention."""
        kept, abstained, reason = apply_threshold([_cand("a", 0.1)], threshold=0.5)
        assert abstained
        assert kept == []
        assert reason

    def test_threshold_is_inclusive(self):
        """A score exactly at the threshold passes; the gate is `>=`, not `>`."""
        kept, abstained, _ = apply_threshold([_cand("a", 0.5)], threshold=0.5)
        assert not abstained
        assert len(kept) == 1

    def test_empty_candidates_abstain(self):
        kept, abstained, reason = apply_threshold([], threshold=0.0)
        assert abstained
        assert kept == []
        assert "no candidates" in reason

    def test_only_top_score_is_tested(self):
        """The gate asks whether the best passage is good enough to answer from.

        Filtering every candidate by the same bar would discard a strong supporting
        passage whenever the top one barely passed, which is the common case for a
        question whose answer spans two chunks.
        """
        kept, abstained, _ = apply_threshold(
            [_cand("a", 0.9), _cand("b", 0.2)], threshold=0.5
        )
        assert not abstained
        assert len(kept) == 2

    def test_oversized_chunk_is_dropped_but_others_survive(self):
        """A chunk larger than MAX_CHUNK_TOKENS is unciteable and is dropped.

        Dropping one passage must not abort the query when other evidence remains.
        """
        kept, abstained, _ = apply_threshold(
            [_cand("big", 0.9, tokens=MAX_CHUNK_TOKENS + 1), _cand("ok", 0.8, tokens=50)],
            threshold=0.5,
        )
        assert not abstained
        assert [c.chunk_id for c in kept] == ["ok"]

    def test_abstains_when_only_candidate_is_oversized(self):
        """The gate was passed by the top candidate, but nothing is left to use."""
        kept, abstained, reason = apply_threshold(
            [_cand("a", 0.9, tokens=MAX_CHUNK_TOKENS + 1)], threshold=0.5
        )
        assert abstained
        assert kept == []
        assert "token cap" in reason

    def test_reason_names_the_scores(self):
        """The abstain reason is logged verbatim; a vague one is undiagnosable."""
        _, _, reason = apply_threshold([_cand("a", 0.11)], threshold=0.5)
        assert "0.1100" in reason
        assert "0.5000" in reason

    def test_oversized_only_candidate_abstains_with_a_token_cap_reason(self):
        kept, abstained, reason = apply_threshold(
            [_cand("a", 0.9, tokens=MAX_CHUNK_TOKENS + 1)], threshold=0.5
        )
        assert abstained
        assert kept == []
        assert "token cap" in reason

    def test_oversized_chunks_never_reach_the_budget(self):
        """An unciteable chunk must be removed before capacity is measured.

        Letting a 50k-token passage into the budget stage would consume the whole
        window on its own and evict every usable passage with it.
        """
        kept, _, _ = apply_threshold(
            [_cand("big", 0.9, tokens=MAX_CHUNK_TOKENS + 1), _cand("ok", 0.8, tokens=50)],
            threshold=0.5,
        )
        assert all(c.token_count <= MAX_CHUNK_TOKENS for c in kept)


class TestApplyTokenBudget:
    def test_keeps_everything_that_fits(self):
        kept, evicted = apply_token_budget([_cand("a", 0.9, tokens=100)], max_context_tokens=1000)
        assert len(kept) == 1
        assert evicted == 0

    def test_evicts_lowest_scoring_to_fit(self):
        """FR-15: highest-ranked passages must survive eviction."""
        cands = [_cand("a", 0.9, tokens=400), _cand("b", 0.5, tokens=400)]
        kept, evicted = apply_token_budget(cands, max_context_tokens=500)
        assert [c.chunk_id for c in kept] == ["a"]
        assert evicted == 1

    def test_answer_reserve_is_held_back(self):
        """A budget counting only context lets the model run out mid-answer."""
        kept, _ = apply_token_budget(
            [_cand("a", 0.9, tokens=800)], max_context_tokens=1000, reserve_for_answer=500
        )
        assert kept == []

    def test_returns_everything_when_budget_is_generous(self):
        cands = [_cand("a", 0.9, tokens=10), _cand("b", 0.8, tokens=10)]
        kept, evicted = apply_token_budget(cands, max_context_tokens=10000)
        assert len(kept) == 2
        assert evicted == 0

    def test_zero_budget_keeps_nothing(self):
        kept, evicted = apply_token_budget([_cand("a", 0.9)], max_context_tokens=0)
        assert kept == []
        assert evicted == 1

    def test_evidence_gaps_are_skipped_not_fatal(self):
        """One oversized chunk must not prevent smaller ones fitting after it."""
        cands = [_cand("big", 0.9, tokens=900), _cand("small", 0.8, tokens=50)]
        kept, evicted = apply_token_budget(cands, max_context_tokens=500)
        assert [c.chunk_id for c in kept] == ["small"]
        assert evicted == 1

    def test_falls_back_to_estimated_tokens_when_unset(self):
        """Rows written before `token_count` was populated still need a budget."""
        cand = RetrievalCandidate(chunk_id="a", doc_id="d", chunk_index=0, text="x" * 400)
        kept, _ = apply_token_budget([cand], max_context_tokens=50)
        assert kept == []


class TestScoreSelection:
    def test_prefers_reranked_score(self):
        c = _cand("a", 0.8)
        c.scores[Stage.FUSED] = 0.03
        assert score_for_threshold([c]) == pytest.approx(0.8)

    def test_falls_back_to_fused_score(self):
        c = RetrievalCandidate(chunk_id="a", doc_id="d", chunk_index=0, text="t")
        c.scores[Stage.FUSED] = 0.03
        assert score_for_threshold([c]) == pytest.approx(0.03)

    def test_empty_returns_zero(self):
        assert score_for_threshold([]) == 0.0

    def test_stage_of_score_reports_the_gating_stage(self):
        """The eval harness must agree with the pipeline on which score was gated."""
        c = _cand("a", 0.8)
        assert stage_of_score(c) is Stage.RERANKED

        d = RetrievalCandidate(chunk_id="b", doc_id="d", chunk_index=1, text="t")
        d.scores[Stage.VECTOR] = 0.5
        assert stage_of_score(d) is Stage.VECTOR

    def test_stage_of_score_none_when_unscored(self):
        c = RetrievalCandidate(chunk_id="c", doc_id="d", chunk_index=2, text="t")
        assert stage_of_score(c) is None


class TestEstimateTokens:
    def test_roughly_four_characters_per_token(self):
        assert estimate_tokens("a" * 400) == 100

    def test_empty_text_is_one_token_not_zero(self):
        """Zero would let a chunk cost nothing and flood the context window."""
        assert estimate_tokens("") == 1