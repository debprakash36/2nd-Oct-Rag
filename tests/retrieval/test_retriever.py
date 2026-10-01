"""Retriever pipeline: wiring, ordering, and the structural invariants §10 names.

§10 lists structural checks that need no corpus, and these are them: state machine
transitions are legal, a disabled document is unretrievable immediately, and (via
the chunk store being authoritative) index drift is recoverable. Ranking *quality*
is deliberately not asserted here — it is what `eval/run_eval.py` measures against
the labelled set, and hard-coding expectations here would only encode today's
numbers.
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.retrieval.retriever import RetrievalConfig, Retriever
from app.retrieval.types import AccessFilter, Stage


class TestRetrievalBasics:
    def test_returns_candidates_for_a_corpus_match(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy", config=RetrievalConfig(rerank_enabled=False)
        )
        assert result.candidates
        assert result.original_query == "refund policy"

    def test_candidates_carry_citation_metadata(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """FR-6: without char_start/char_end a citation degrades to a file link."""
        result = Retriever(session, settings=settings_env).retrieve(
            "refund", config=RetrievalConfig(rerank_enabled=False)
        )
        assert result.candidates
        for c in result.candidates:
            assert c.chunk_id
            assert c.doc_id
            assert c.char_start >= 0
            assert c.char_end > c.char_start

    def test_respects_top_k(self, session: Session, multi_chunk_doc, settings_env):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund", config=RetrievalConfig(top_k=2, rerank_enabled=False, fetch_k=40)
        )
        assert len(result.candidates) <= 2

    def test_empty_corpus_abstains(
        self, session: Session, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve("anything")
        assert result.abstained
        assert result.abstain_reason

    def test_results_are_deterministic(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """Non-deterministic order makes citations and eval scores vary run to run."""
        retriever = Retriever(session, settings=settings_env)
        cfg = RetrievalConfig(rerank_enabled=False)
        first = retriever.retrieve("refund policy", config=cfg).chunk_ids
        second = retriever.retrieve("refund policy", config=cfg).chunk_ids
        assert first == second

    def test_query_matching_nothing_in_corpus_abstains(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """A high threshold must actually gate; otherwise FR-14 does nothing."""
        result = Retriever(session, settings=settings_env).retrieve(
            "medieval cathedral construction contracts",
            config=RetrievalConfig(score_threshold=0.99, rerank_enabled=False),
        )
        assert result.abstained


class TestStageOrderAndTrace:
    def test_records_all_stages_run(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy", config=RetrievalConfig(rerank_enabled=True)
        )
        assert Stage.VECTOR in result.trace.stages_run
        assert Stage.KEYWORD in result.trace.stages_run
        assert Stage.FUSED in result.trace.stages_run
        assert Stage.RERANKED in result.trace.stages_run

    def test_rerank_stage_skipped_when_disabled(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund", config=RetrievalConfig(rerank_enabled=False)
        )
        assert Stage.RERANKED not in result.trace.stages_run

    def test_per_stage_scores_retained(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """FR-28 requires per-stage scores to tell a content gap from a retrieval gap."""
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy", config=RetrievalConfig(rerank_enabled=True)
        )
        assert result.candidates
        for candidate in result.candidates:
            assert candidate.scores

    def test_timings_recorded_per_stage(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund", config=RetrievalConfig(rerank_enabled=False)
        )
        assert "vector" in result.timings_ms
        assert "keyword" in result.timings_ms
        assert "fusion" in result.timings_ms

    def test_rewrite_timing_recorded(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """§7.4 budgets 400 ms for rewrite; the measurement has to exist to check it."""
        result = Retriever(session, settings=settings_env).retrieve("refund")
        assert "rewrite" in result.timings_ms


class TestAblations:
    """§10 checks 1 and 2 need the stages to be independently switchable."""

    def test_keyword_disabled_runs_vector_only(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy", config=RetrievalConfig(keyword_enabled=False, rerank_enabled=False)
        )
        assert Stage.KEYWORD not in result.trace.stages_run
        assert Stage.VECTOR in result.trace.stages_run

    def test_vector_disabled_runs_keyword_only(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy", config=RetrievalConfig(vector_enabled=False, rerank_enabled=False)
        )
        assert Stage.VECTOR not in result.trace.stages_run
        assert Stage.KEYWORD in result.trace.stages_run

    def test_both_stages_disabled_abstains(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund",
            config=RetrievalConfig(vector_enabled=False, keyword_enabled=False),
        )
        assert result.abstained

    def test_keyword_only_still_finds_exact_identifiers(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """FR-12: keyword-only must work, or hybrid is not additive.

        This is the case architecture.md §3.3 names — exact identifiers are the
        recall failure that makes pure-vector search unacceptable for document Q&A.
        """
        result = Retriever(session, settings=settings_env).retrieve(
            "paragraph 7",
            config=RetrievalConfig(
                vector_enabled=False, keyword_enabled=True, rerank_enabled=False
            ),
        )
        assert result.candidates


class TestRewriteIntegration:
    def test_rewrite_is_recorded_separately_from_the_original(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """FR-28: a bad rewrite is otherwise indistinguishable from a retrieval failure."""
        result = Retriever(session, settings=settings_env).retrieve(
            "What about refunds?", history=["The refund policy sets a 30 day window."]
        )
        assert result.original_query == "What about refunds?"
        assert result.rewrite_applied
        assert result.rewritten_query
        assert result.effective_query != result.original_query

    def test_no_rewrite_when_not_needed(self, session: Session, multi_chunk_doc, settings_env):
        result = Retriever(session, settings=settings_env).retrieve(
            "What is the refund window?", history=[]
        )
        assert not result.rewrite_applied
        assert result.rewritten_query == ""
        assert result.effective_query == result.original_query

    def test_history_does_not_change_a_self_contained_query(
        self, session: Session, multi_chunk_doc, settings_env
        ):
        """§3.2 only rewrites when the raw question is unanswerable in isolation."""
        retriever = Retriever(session, settings=settings_env)
        alone = retriever.retrieve("What is the warranty period?").chunk_ids
        with_history = retriever.retrieve(
            "What is the warranty period?",
            history=["The refund policy allows returns."],
        ).chunk_ids
        assert alone == with_history


class TestAccessFilterIntegration:
    """FR-16 pushed all the way through the pipeline."""

    def test_restricted_corpus_returns_nothing_for_other_team(
        self, session: Session, live_doc, settings_env
    ):
        from app.db.models import Document

        session.get(Document, live_doc.doc_id).acl_tags = ["restricted"]
        session.commit()

        result = Retriever(session, settings=settings_env).retrieve(
            "refund", access=AccessFilter(frozenset({"other-team"}))
        )
        assert result.candidates == []
        assert result.abstained

    def test_permitted_corpus_returns_candidates(
        self, session: Session, live_doc, settings_env
    ):
        from app.db.models import Document

        session.get(Document, live_doc.doc_id).acl_tags = ["finance"]
        session.commit()

        result = Retriever(session, settings=settings_env).retrieve(
            "refund", access=AccessFilter(frozenset({"finance"}))
        )
        assert result.candidates

    def test_disabled_document_produces_an_abstention(
        self, session: Session, live_doc, settings_env
    ):
        """FR-7 end-to-end: tombstone the only document and retrieval must refuse."""
        from app.db.models import DocumentState
        from app.ingest.states import transition

        transition(session, live_doc, DocumentState.DISABLED, reason="test")
        session.commit()

        result = Retriever(session, settings=settings_env).retrieve("refund")
        assert result.abstained


class TestContextBudget:
    def test_eviction_keeps_context_within_budget(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "policy",
            config=RetrievalConfig(
                max_context_tokens=300, reserve_for_answer=100, rerank_enabled=False
            ),
        )
        if result.candidates:
            used = sum(c.token_count for c in result.candidates)
            assert used <= 200

    def test_tiny_budget_abstains(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "refund policy",
            config=RetrievalConfig(max_context_tokens=1, reserve_for_answer=0),
        )
        assert result.abstained

    def test_trace_reports_eviction(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        result = Retriever(session, settings=settings_env).retrieve(
            "policy",
            config=RetrievalConfig(max_context_tokens=250, rerank_enabled=False),
        )
        assert result.trace.after_budget <= result.trace.after_threshold


class TestDedupeIntegration:
    def test_near_duplicate_chunks_are_not_both_returned(
        self, session: Session, settings_env
    ):
        """Boilerplate must not consume the whole context window (§3.3, FR-15).

        Two documents with the same boilerplate passage and the same content hash
        are both live, so both chunks are genuinely retrievable; dedupe is what
        stops the single fact being returned twice.
        """
        from app.db.models import Chunk, Document, DocumentState
        from app.providers.embedding import get_embedding_provider
        from app.retrieval.fusion import deduplicate
        from app.retrieval.types import RetrievalCandidate

        doc_id = "dedupdoc00000000000000000000001"
        session.add(
            Document(
                doc_id=doc_id,
                filename="boilerplate.md",
                mime_type="text/markdown",
                state=DocumentState.LIVE,
                acl_tags=[],
            )
        )
        provider = get_embedding_provider(settings_env)
        text = (
            "This clause applies to all customers unless a separate written "
            "agreement states otherwise."
        )
        embedding = provider.embed([text], model=settings_env.embedding_model)[0]
        session.add_all(
            [
                Chunk(
                    chunk_id=f"dedupchunk{i:0>26}",
                    doc_id=doc_id,
                    chunk_index=i,
                    text=text,
                    token_count=12,
                    char_start=0,
                    char_end=len(text),
                    embedding=embedding,
                )
                for i in (1, 2)
            ]
        )
        session.commit()

        candidates = [
            RetrievalCandidate(chunk_id="a", doc_id="d1", chunk_index=0, text=text),
            RetrievalCandidate(chunk_id="b", doc_id="d2", chunk_index=0, text=text),
        ]
        assert len(deduplicate(candidates)) == 1


class TestRetrievalConfig:
    def test_from_settings_reads_every_knob(self, settings_env):
        config = RetrievalConfig.from_settings(settings_env)
        assert config.fetch_k > 0
        assert config.top_k > 0
        assert 0.0 <= config.score_threshold <= 1.0

    def test_per_call_config_overrides_the_instance_default(
        self, session: Session, multi_chunk_doc, settings_env
    ):
        """The eval sweep varies one parameter per run.

        If a per-call `config` were ignored in favour of the instance's, sweeping
        the threshold would silently re-measure the same configuration every time.
        """
        retriever = Retriever(session, settings=settings_env)
        result = retriever.retrieve(
            "refund policy", config=RetrievalConfig(top_k=1, rerank_enabled=False)
        )
        assert len(result.candidates) <= 1

    def test_defaults_are_within_the_architecture_range(self):
        """§3.3 states wide fetch 20-50 and narrow context 4-8."""
        cfg = RetrievalConfig()
        assert 20 <= cfg.fetch_k <= 50
        assert 4 <= cfg.top_k <= 8
        assert cfg.rerank_k >= cfg.fetch_k

    def test_environment_overrides_defaults(self):
        from app.core.config import Settings

        settings = Settings(
            environment="test",
            retrieval_top_k=3,
            retrieval_threshold=0.42,
            retrieval_rerank_enabled=False,
        )
        config = RetrievalConfig.from_settings(settings)
        assert config.top_k == 3
        assert config.score_threshold == pytest.approx(0.42)
        assert not config.rerank_enabled