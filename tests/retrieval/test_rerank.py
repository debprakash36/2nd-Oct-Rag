"""Rerank stage and providers (FR-13).

Two things are worth separating here. The *stage* logic in
`app/retrieval/rerank.py` is about correctly mapping provider scores back onto
candidates. The *providers* are about score quality, and neither of them is a
trained cross-encoder — see the caveat in both modules.
"""

from __future__ import annotations

import pytest

from app.providers.base import ScoredDoc
from app.providers.rerank import (
    LexicalRerankerProvider,
    NullRerankerProvider,
    get_reranker_provider,
)
from app.retrieval.rerank import rerank_candidates
from app.retrieval.types import RetrievalCandidate, Stage


def _cand(chunk_id: str, text: str, *, doc_id: str = "d1", index: int = 0) -> RetrievalCandidate:
    return RetrievalCandidate(chunk_id=chunk_id, doc_id=doc_id, chunk_index=index, text=text)


class _StubReranker:
    """Reranker returning predetermined scores, to test the stage not a provider."""

    def __init__(self, scores: list[float] | None = None, *, return_all: bool = True) -> None:
        self._scores = scores
        self._return_all = return_all
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, docs, *, top_k: int) -> list[ScoredDoc]:
        self.calls.append((query, list(docs), top_k))
        if not self._return_all:
            docs = list(docs)[:top_k]
        scores = self._scores or [1.0 / (i + 1) for i in range(len(docs))]
        return [
            ScoredDoc(chunk_id=str(i), score=scores[i] if i < len(scores) else 0.0, text=d)
            for i, d in enumerate(docs)
        ]


class TestRerankStage:
    def test_sets_reranked_score_on_each_candidate(self):
        out = rerank_candidates(
            "refund", [_cand("a", "alpha"), _cand("b", "beta")], _StubReranker(), top_k=10
        )
        assert all(Stage.RERANKED in c.scores for c in out)

    def test_sorts_by_descending_rerank_score(self):
        out = rerank_candidates(
            "refund",
            [_cand("a", "alpha"), _cand("b", "beta"), _cand("c", "gamma")],
            _StubReranker(scores=[0.1, 0.9, 0.5]),
            top_k=10,
        )
        assert [c.chunk_id for c in out] == ["b", "c", "a"]

    def test_assigns_one_based_ranks(self):
        out = rerank_candidates(
            "refund",
            [_cand("a", "alpha"), _cand("b", "beta")],
            _StubReranker(scores=[0.3, 0.7]),
            top_k=10,
        )
        assert [c.ranks[Stage.RERANKED] for c in out] == [1, 2]

    def test_scores_map_to_the_right_chunk(self):
        """Score-to-candidate mapping is by text, not by provider's positional id.

        `ScoredDoc.chunk_id` is the *provider's* label, not our chunk_id. Trusting
        it would attach a score to whatever chunk happened to occupy that index.
        """
        out = rerank_candidates(
            "q",
            [_cand("first", "about refunds"), _cand("second", "about shipping")],
            _StubReranker(scores=[0.9, 0.1]),
            top_k=10,
        )
        assert out[0].chunk_id == "first"
        assert out[0].scores[Stage.RERANKED] == pytest.approx(0.9)

    def test_partial_results_leave_remainder_unranked(self):
        """A provider returning fewer results must not shift scores onto wrong chunks.

        If the reranker returns only the first candidate, the second is dropped
        rather than inheriting a score. Dropping is the safe direction: an
        unranked passage could otherwise reach the model with a fabricated score.
        """
        out = rerank_candidates(
            "q",
            [_cand("a", "alpha"), _cand("b", "beta"), _cand("c", "gamma")],
            _StubReranker(return_all=False),
            top_k=1,
        )
        assert len(out) == 1

    def test_empty_candidates_returns_empty(self):
        assert rerank_candidates("q", [], _StubReranker(), top_k=10) == []

    def test_reranker_returning_nothing_yields_nothing(self):
        """A provider failure is not an empty corpus; the retriever abstains on it."""
        class _Empty:
            def rerank(self, query, docs, *, top_k):
                return []

        assert rerank_candidates("q", [_cand("a", "alpha")], _Empty(), top_k=10) == []

    def test_top_k_is_passed_through(self):
        stub = _StubReranker()
        rerank_candidates("q", [_cand("a", "alpha")], stub, top_k=7)
        assert stub.calls[0][2] == 7


class TestLexicalRerankerProvider:
    def setup_method(self):
        self.provider = LexicalRerankerProvider()

    def test_relevant_document_scores_above_irrelevant(self):
        scored = self.provider.rerank(
            "refund policy",
            ["The refund policy allows returns within 30 days.", "Cats sleep a lot."],
            top_k=2,
        )
        assert scored[0].text.startswith("The refund")

    def test_scores_are_bounded_zero_to_one(self):
        """A bounded scale keeps `RETRIEVAL_THRESHOLD` interpretable.

        An unbounded score would make the threshold's meaning depend on the
        provider, and a threshold calibrated against one provider would silently
        stop abstaining with another.
        """
        scored = self.provider.rerank(
            "refund", ["refund refund refund refund refund"], top_k=1
        )
        assert 0.0 <= scored[0].score <= 1.0

    def test_stopword_only_query_scores_zero(self):
        """No signal in the query means no invented preference."""
        scored = self.provider.rerank("the of and", ["anything at all"], top_k=1)
        assert scored[0].score == 0.0

    def test_adjacent_terms_beat_separated_terms(self):
        """Adjacency bonus: `refund window` together beats the same terms apart.

        A bag-of-words overlap count cannot express this, and it is the difference
        between a chunk about the refund *window* and one that mentions both words
        in unrelated sections.
        """
        together = "The refund window is thirty days from purchase."
        apart = "A refund may apply. " + ("Filler sentence. " * 40) + "The window closes."
        scored = self.provider.rerank("refund window", [apart, together], top_k=2)
        assert scored[0].text == together

    def test_longer_document_is_not_heavily_penalised(self):
        """Length normalisation is logarithmic, so a long relevant chunk still wins."""
        long_doc = "refund " + ("filler text " * 200) + "refund policy window"
        short_irrelevant = "Unrelated content entirely."
        scored = self.provider.rerank("refund window", [short_irrelevant, long_doc], top_k=2)
        assert scored[0].text == long_doc

    def test_respects_top_k(self):
        docs = [f"document {i} about refunds" for i in range(10)]
        assert len(self.provider.rerank("refund", docs, top_k=3)) == 3

    def test_empty_docs_handled(self):
        assert self.provider.rerank("refund", [], top_k=5) == []

    def test_empty_document_scores_zero(self):
        scored = self.provider.rerank("refund", [""], top_k=1)
        assert scored[0].score == 0.0


class TestNullRerankerProvider:
    def test_scores_everything_zero(self):
        """Zero scores are below any positive threshold, so the pipeline abstains.

        That is the intended failure direction (architecture.md §7.3): no answer
        rather than an unranked one.
        """
        scored = NullRerankerProvider().rerank("q", ["a", "b"], top_k=2)
        assert [s.score for s in scored] == [0.0, 0.0]

    def test_passes_text_through(self):
        scored = NullRerankerProvider().rerank("q", ["alpha", "beta"], top_k=2)
        assert [s.text for s in scored] == ["alpha", "beta"]


class TestGetRerankerProvider:
    def test_default_is_lexical(self, settings_env):
        assert isinstance(get_reranker_provider(settings_env), LexicalRerankerProvider)

    def test_null_selected_by_config(self):
        from app.core.config import Settings

        settings = Settings(environment="test", reranker_provider="null")
        assert isinstance(get_reranker_provider(settings), NullRerankerProvider)

    def test_satisfies_the_protocol(self, settings_env):
        """The NFR-10 seam: callers depend on the interface, not the class."""
        from app.providers.base import RerankerProvider

        assert isinstance(get_reranker_provider(settings_env), RerankerProvider)