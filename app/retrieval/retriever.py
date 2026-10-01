"""The retriever: architecture.md §3.3 assembled into one callable object.

Stage order is the architecture's, and the reason for the ordering is that each
stage is cheaper to reject on than the next:

    rewrite -> [vector ‖ keyword] -> RRF fusion -> dedupe -> rerank -> threshold -> budget

Vector and keyword search run **in parallel** against the same logical corpus
(§3.3). They are independent reads of the same rows, so there is no reason to
serialise them, and §7.4 budgets them together at 150 ms.

Rerank runs after fusion, not inside each stage, because a cross-encoder is
roughly two orders of magnitude more expensive per candidate than a vector
distance. Running it on the union of both candidate lists would cost 2x for the
same ranking. It is toggleable so §10 check 2 can A/B it.

`RetrievalConfig` carries the tunable numbers. They are config rather than
literals precisely because architecture.md §3.3 calls the threshold a *measured*
parameter: it has to be swept by the eval harness without a code change.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import StoreUnavailableError
from app.db.session import get_session_factory
from app.providers.base import EmbeddingProvider, RerankerProvider
from app.providers.embedding import get_embedding_provider
from app.providers.rerank import get_reranker_provider
from app.retrieval.fusion import RRF_K, deduplicate, reciprocal_rank_fusion
from app.retrieval.keyword_index import KeywordIndex, build_keyword_index
from app.retrieval.rerank import rerank_candidates
from app.retrieval.rewrite import rewrite_query
from app.retrieval.threshold import apply_threshold, apply_token_budget
from app.retrieval.types import (
    AccessFilter,
    RetrievalCandidate,
    RetrievalResult,
    Stage,
)
from app.retrieval.vector_store import VectorStore, build_vector_store

log = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Tunable retrieval parameters.

    Defaults are the architecture.md §3.3 starting points: wide fetch 20-50,
    narrow final context 4-8. `overfetch` is 5x the fetch size, the common
    convention for giving a reranker room to reorder.
    """

    #: Wide fetch per stage (§3.3).
    fetch_k: int = 40
    #: Rerank depth before narrowing.
    rerank_k: int = 40
    #: Narrow final context (§3.3).
    top_k: int = 8
    #: Relevance gate (FR-14). Calibrated by the eval harness, not guessed.
    score_threshold: float = 0.0
    #: Hard cap on retrieved context tokens (FR-15).
    max_context_tokens: int = 4000
    #: Tokens held back for the generated answer.
    reserve_for_answer: int = 500
    #: Enable the cross-encoder stage (FR-13).
    rerank_enabled: bool = True
    #: Run the keyword half of hybrid search (FR-12).
    keyword_enabled: bool = True
    #: Run the vector half of hybrid search.
    vector_enabled: bool = True
    #: RRF damping constant.
    rrf_k: int = RRF_K
    #: Optional per-stage fusion weights. Unset means equal weight, which is the
    #: RRF default and the honest choice until the eval set says otherwise.
    fusion_weights: dict[Stage, float] | None = None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> RetrievalConfig:
        """Build a config from application settings.

        Every retrieval knob is overridable from the environment so the eval
        harness can sweep it without editing code (architecture.md §3.3).
        """
        s = settings or get_settings()
        return cls(
            fetch_k=s.retrieval_fetch_k,
            rerank_k=s.retrieval_rerank_k,
            top_k=s.retrieval_top_k,
            score_threshold=s.retrieval_threshold,
            max_context_tokens=s.retrieval_max_context_tokens,
            reserve_for_answer=s.retrieval_reserve_for_answer,
            rerank_enabled=s.retrieval_rerank_enabled,
            keyword_enabled=s.retrieval_keyword_enabled,
            vector_enabled=s.retrieval_vector_enabled,
        )

    def for_measurement(self, depth: int) -> RetrievalConfig:
        """A copy whose final list is `depth` deep, for recall@k measurement.

        Recall@k is a property of the **ranking**, so it has to be measured over
        a list at least k long. `top_k` is the architecture's narrow *context*
        budget (§3.3, 4-8 chunks) and is deliberately much smaller than 10, so
        scoring recall@10 against `result.candidates` cannot exceed
        `top_k / n` -- it was structurally capped at 0.80 and could never reach
        the 0.85 gate no matter how well retrieval performed.

        The eval harness widens the returned list only. `max_context_tokens` is
        left alone: it is a real capacity limit on what the model will be given,
        so silently raising it would let a question score a hit on a passage the
        generator would never have received. What that does mean is that recall@k
        can exceed the count the model actually sees, which is why the harness
        reports both numbers rather than only the flattering one.
        """
        import dataclasses

        if depth <= self.top_k:
            return self
        return dataclasses.replace(self, top_k=depth)


class Retriever:
    """Hybrid retriever. One instance per request; it holds no mutable state."""

    def __init__(
        self,
        session: Session,
        *,
        vector_store: VectorStore | None = None,
        keyword_index: KeywordIndex | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: RerankerProvider | None = None,
        config: RetrievalConfig | None = None,
        settings: Settings | None = None,
        glossary: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Dependencies are injectable so tests and the eval harness can supply
        a store or provider without touching the settings or the database."""
        self._session = session
        self._settings = settings or get_settings()
        self._vector_store = vector_store or build_vector_store(session, self._settings)
        self._keyword_index = keyword_index or build_keyword_index(session)
        self._embeddings = embedding_provider or get_embedding_provider(self._settings)
        self._reranker = reranker or get_reranker_provider(self._settings)
        self._config = config or RetrievalConfig.from_settings(self._settings)
        # Settings-owned glossary is the default so a deployed value applies
        # without every call site passing one; an explicit argument wins. Typed
        # loosely because `Settings.glossary` yields tuples and callers supply
        # lists; both are valid `Sequence[str]` synonym values.
        self._glossary: Mapping[str, Sequence[str]] | None = (
            glossary if glossary is not None else self._settings.glossary
        )

    def _new_search_session(self) -> Session:
        """A short-lived read session for one search stage.

        Separate from the request session because `Session` is not thread-safe and
        the two search stages run concurrently. The session is closed by the store
        once the search returns -- see the `with` in each stage below.

        Using the same engine rather than a new one keeps the connection pool
        shared, so this does not open a second pool per query.
        """
        return get_session_factory(self._settings)()

    # -- stages ------------------------------------------------------------

    def _vector_candidates(
        self,
        query: str,
        access: AccessFilter | None,
        session: Session,
    ) -> tuple[list[RetrievalCandidate], float]:
        """Stage 1a. Takes its own session.

        Both search stages issue SQL, and a `Session` is not safe for concurrent
        use -- SQLAlchemy raises `InvalidRequestError` when two threads share one.
        The request thread keeps its session for everything else, so each stage
        opens its own short-lived read session instead of sharing.
        """
        start = time.perf_counter()
        with session:
            vector = self._embeddings.embed(
                [query], model=self._settings.embedding_model
            )[0]
            store = build_vector_store(session, self._settings)
            hits = store.search(vector, k=self._config.fetch_k, access=access)
        return hits, (time.perf_counter() - start) * 1000

    def _keyword_candidates(
        self,
        query: str,
        access: AccessFilter | None,
        session: Session,
    ) -> tuple[list[RetrievalCandidate], float]:
        """Stage 1b. Takes its own session, for the reason in `_vector_candidates`."""
        start = time.perf_counter()
        with session:
            index = build_keyword_index(session)
            hits = index.search(query, k=self._config.fetch_k, access=access)
        return hits, (time.perf_counter() - start) * 1000

    # -- entry point -------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        history: Sequence[str] | None = None,
        access: AccessFilter | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        """Retrieve candidates for `query`.

        Returns a `RetrievalResult` rather than a bare list because abstention and
        the per-stage trace are part of the answer: FR-28 logs both the original
        and rewritten query, and the eval harness attributes a miss to a stage.
        """
        cfg = config or self._config
        result = RetrievalResult(original_query=query)
        trace = result.trace
        timings = result.timings_ms

        if not self._config.vector_enabled and not self._config.keyword_enabled:
            result.abstained = True
            result.abstain_reason = "both retrieval stages disabled"
            log.warning("retrieval invoked with all stages disabled")
            return result

        # Stage 0: rewrite. Timed because §7.4 budgets it at 400 ms and it is the
        # stage most likely to regress when a provider is added.
        rewrite_start = time.perf_counter()
        rewrite = rewrite_query(query, history, glossary=self._glossary)
        timings["rewrite"] = (time.perf_counter() - rewrite_start) * 1000
        result.rewritten_query = rewrite.query if rewrite.changed else ""
        result.rewrite_applied = rewrite.changed
        search_query = rewrite.query

        if not search_query.strip():
            result.abstained = True
            result.abstain_reason = "empty query"
            return result

        # Stage 1: wide fetch, vector and keyword in parallel (§3.3). Each stage
        # is submitted independently so disabling one still runs the other; a
        # nested submission would silently turn "keyword only" into "no search".
        stages: dict[Stage, Sequence[RetrievalCandidate]] = {}
        enabled = [stage for stage, on in (
            (Stage.VECTOR, cfg.vector_enabled),
            (Stage.KEYWORD, cfg.keyword_enabled),
        ) if on]

        if enabled:
            with ThreadPoolExecutor(
                max_workers=len(enabled), thread_name_prefix="retrieve"
            ) as pool:
                futures = {}
                for stage in enabled:
                    fn = (
                        self._vector_candidates
                        if stage is Stage.VECTOR
                        else self._keyword_candidates
                    )
                    futures[stage] = pool.submit(
                        fn, search_query, access, self._new_search_session()
                    )
                for stage, future in futures.items():
                    try:
                        hits, elapsed = future.result()
                    except StoreUnavailableError:
                        raise
                    except SQLAlchemyError as exc:
                        # A search stage cannot reach its store. This is not "no
                        # matches" and must never be reported as such -- see
                        # `StoreUnavailableError`.
                        raise StoreUnavailableError(stage.value) from exc
                    timings[stage.value] = elapsed
                    stages[stage] = hits
                    trace.record(stage, len(hits))

        # Stage 2: fusion.
        fusion_start = time.perf_counter()
        fused = reciprocal_rank_fusion(
            stages, rrf_k=cfg.rrf_k, weights=cfg.fusion_weights
        )
        timings["fusion"] = (time.perf_counter() - fusion_start) * 1000
        trace.record(Stage.FUSED, len(fused))

        # Dedupe before rerank: identical boilerplate burns reranker calls and
        # context slots, so removing it first is strictly cheaper (§3.3).
        deduped = deduplicate(fused)
        trace.after_dedupe = len(deduped)

        if not deduped:
            result.abstained = True
            result.abstain_reason = "no candidates survived fusion and dedupe"
            result.timings_ms = timings
            return result

        # Stage 3: rerank (FR-13), then narrow.
        candidates = deduped
        if cfg.rerank_enabled and deduped:
            rerank_start = time.perf_counter()
            candidates = rerank_candidates(
                search_query,
                deduped[: cfg.rerank_k],
                self._reranker,
                top_k=cfg.rerank_k,
            )
            timings["rerank"] = (time.perf_counter() - rerank_start) * 1000
            trace.record(Stage.RERANKED, len(candidates))
            if not candidates:
                # A reranker returning nothing is a provider failure, not an
                # empty corpus. Abstain rather than fall through to unranked
                # candidates, which is the direction §7.3 warns about.
                result.abstained = True
                result.abstain_reason = "reranker returned no candidates"
                result.timings_ms = timings
                return result

        # Stage 4: threshold, then budget.
        threshold_start = time.perf_counter()
        kept, abstained, reason = apply_threshold(candidates, cfg.score_threshold)
        timings["threshold"] = (time.perf_counter() - threshold_start) * 1000
        result.threshold_applied = cfg.score_threshold
        trace.after_threshold = len(kept)

        if abstained:
            result.abstained = True
            result.abstain_reason = reason
            # Retained for the sources panel only (FR-20). `candidates` stays
            # empty so recall measurement is unaffected by below-threshold rows.
            result.abstain_candidates = candidates[: cfg.top_k]
            result.timings_ms = timings
            log.info(
                "retrieval abstained",
                extra={"reason": reason, "query_len": len(search_query)},
            )
            return result

        budget_start = time.perf_counter()
        final, evicted = apply_token_budget(
            kept,
            cfg.max_context_tokens,
            reserve_for_answer=cfg.reserve_for_answer,
        )
        timings["budget"] = (time.perf_counter() - budget_start) * 1000
        trace.after_budget = len(final)

        if not final:
            result.abstained = True
            result.abstain_reason = "context budget too small for any candidate"
            result.abstain_candidates = kept[: cfg.top_k]
            result.timings_ms = timings
            return result

        result.candidates = final[: cfg.top_k]
        result.timings_ms = timings
        log.debug(
            "retrieval complete",
            extra={
                "returned": len(result.candidates),
                "evicted": evicted,
                "rewrite_applied": result.rewrite_applied,
                "total_ms": round(sum(timings.values()), 2),
            },
        )
        return result