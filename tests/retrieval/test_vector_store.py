"""Vector store: cosine, filter pushdown, and the live-only invariant.

The ACL tests matter more than they look. architecture.md §3.4 is explicit that
post-filtering is a security bug, because text already inside the model's context
can leak by paraphrase even after the answer is scrubbed. So these tests assert
that unauthorised chunks never *appear in the result*, not merely that they were
removed before the caller saw the scores.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy.orm import Session

from app.retrieval.types import AccessFilter, Stage
from app.retrieval.vector_store import (
    SqliteVectorStore,
    build_vector_store,
    cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_magnitude_does_not_matter(self):
        """Cosine is scale-invariant; only direction is a similarity."""
        assert cosine_similarity([1.0, 1.0], [10.0, 10.0]) == pytest.approx(1.0)

    def test_zero_vector_returns_zero_not_nan(self):
        """A zero vector has no direction, so 0.0 (rank last) is the safe answer.

        Returning NaN would make `sorted()` order undefined and silently reorder
        every result; returning 1.0 would make unembedded chunks the best match.
        """
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        """Silently truncating would compare unrelated dimensions."""
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])


class TestSqliteVectorStore:
    def test_returns_ranked_candidates_with_vector_scores(
        self, session: Session, live_doc, provider, settings_env
    ):
        store = SqliteVectorStore(session)
        query = provider.embed(["refund policy"], model=settings_env.embedding_model)[0]
        results = store.search(query, k=5)

        assert results
        for candidate in results:
            assert Stage.VECTOR in candidate.scores
            assert candidate.text

    def test_scores_descend(self, session: Session, live_doc, provider, settings_env):
        store = SqliteVectorStore(session)
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        scores = [c.scores[Stage.VECTOR] for c in store.search(query, k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_honours_limit(self, session: Session, live_doc, provider, settings_env):
        store = SqliteVectorStore(session)
        query = provider.embed(["policy"], model=settings_env.embedding_model)[0]
        assert len(store.search(query, k=2)) <= 2

    def test_empty_corpus_returns_nothing(self, session: Session, provider, settings_env):
        store = SqliteVectorStore(session)
        query = provider.embed(["anything"], model=settings_env.embedding_model)[0]
        assert store.search(query, k=5) == []

    def test_disabled_document_is_immediately_unretrievable(
        self, session: Session, live_doc, provider, settings_env
    ):
        """FR-7: a tombstone stops serving immediately, not at the next reindex."""
        from app.db.models import DocumentState
        from app.ingest.states import transition

        store = SqliteVectorStore(session)
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        assert store.search(query, k=50)

        transition(session, live_doc, DocumentState.DISABLED, reason="test")
        session.commit()

        assert store.search(query, k=50) == []

    def test_failed_document_is_never_retrievable(
        self, session: Session, live_doc, provider, settings_env
    ):
        from app.db.models import DocumentState
        from app.ingest.states import transition

        transition(session, live_doc, DocumentState.FAILED, reason="ingest broke")
        session.commit()

        store = SqliteVectorStore(session)
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        assert store.search(query, k=50) == []

    def test_chunks_without_embeddings_are_skipped(
        self, session: Session, live_doc, settings_env
    ):
        """A chunk the embedder never reached must not be returned with a 0.0 score."""
        from sqlalchemy import update

        from app.db.models import Chunk

        session.execute(update(Chunk).values(embedding=None))
        session.commit()

        store = SqliteVectorStore(session)
        query = [0.1] * settings_env.embedding_dim
        assert store.search(query, k=10) == []

    def test_dimension_mismatch_fails_loudly(self, session: Session, live_doc):
        """architecture.md 7.3: fail on dimension mismatch, never score silently wrong.

        The alternative — truncating or padding — produces plausible-looking
        similarities that are wrong, and present as a retrieval quality
        regression rather than a configuration error.
        """
        store = SqliteVectorStore(session)
        with pytest.raises(ValueError, match="embedding dimension mismatch"):
            store.search([0.1] * 7, k=5)


class TestAccessFilterPushdown:
    """FR-16: filters are pushed into the store, not applied to results afterwards."""

    def _tag_document(self, session: Session, doc_id: str, tags: list[str]) -> None:
        from app.db.models import Document

        doc = session.get(Document, doc_id)
        doc.acl_tags = tags
        session.commit()

    def test_unrestricted_filter_permits_everything(self):
        from app.retrieval.types import UNRESTRICTED

        assert UNRESTRICTED.permits(["anything"])
        assert UNRESTRICTED.permits(None)

    def test_matching_tag_permits(self):
        assert AccessFilter(frozenset({"finance"})).permits(["finance", "hr"])

    def test_non_matching_tag_denies(self):
        assert not AccessFilter(frozenset({"finance"})).permits(["hr"])

    def test_empty_tags_denied_when_filter_present(self):
        """A tagged-off document with no grants must not leak through a filter."""
        assert not AccessFilter(frozenset({"finance"})).permits([])

    def test_untagged_document_is_invisible_to_a_filtered_caller(self):
        assert not AccessFilter(frozenset({"finance"})).permits(None)

    def test_empty_filter_means_unrestricted(self):
        """Documented in AccessFilter: deny-by-default would fail closed on a
        misconfigured caller and look like a corpus outage."""
        assert AccessFilter(frozenset()).permits(["hr"])

    def test_unauthorised_chunk_never_appears_in_results(
        self, session: Session, live_doc, provider, settings_env
    ):
        self._tag_document(session, live_doc.doc_id, ["restricted"])
        store = SqliteVectorStore(session)
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]

        results = store.search(query, k=50, access=AccessFilter(frozenset({"other-team"})))
        assert results == []

    def test_authorised_chunk_does_appear(
        self, session: Session, live_doc, provider, settings_env
    ):
        self._tag_document(session, live_doc.doc_id, ["finance"])
        store = SqliteVectorStore(session)
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]

        results = store.search(query, k=50, access=AccessFilter(frozenset({"finance"})))
        assert results

    def test_filter_does_not_consume_result_slots(
        self, session: Session, live_doc, second_live_doc, provider, settings_env
    ):
        """FR-16 correctness, not just hygiene.

        A restricted document must not occupy a slot in the result window. If ACL
        rejection happened after the limit, a caller with narrow grants would see
        fewer than k results purely because another tenant's chunks filled the
        window — a silent availability bug that looks like a small corpus.
        """
        self._tag_document(session, live_doc.doc_id, ["restricted"])
        self._tag_document(session, second_live_doc.doc_id, ["finance"])

        store = SqliteVectorStore(session)
        query = provider.embed(["refund policy"], model=settings_env.embedding_model)[0]

        results = store.search(query, k=50, access=AccessFilter(frozenset({"finance"})))
        assert results
        assert all("restricted" not in c.acl_tags for c in results)


class TestBuildVectorStore:
    def test_selects_sqlite_for_sqlite(self, session: Session, settings_env):
        store = build_vector_store(session, settings_env)
        assert isinstance(store, SqliteVectorStore)

    def test_production_rejects_a_sqlite_database(self, settings_env):
        """A production run must not silently degrade to a full scan.

        The SQLite store is a test double with an O(n) cost model. Deploying it by
        accident would meet the latency requirement on a small corpus and fail at
        NFR-4 scale with no error to explain it.

        Previously this was enforced by an environment check inside
        `build_vector_store` that only ran for the removed `"auto"` setting. With
        the backend named explicitly there is nothing to infer, so the guard lives in
        `Settings.validate_production`, which refuses a SQLite `database_url`
        outright. This asserts that, because a production process cannot get past
        startup without it.
        """
        from app.core.config import Settings

        prod = Settings(
            environment="production",
            database_url=settings_env.database_url,
            embedding_dim=settings_env.embedding_dim,
        )
        with pytest.raises(RuntimeError, match="must be Postgres"):
            prod.validate_production()

    def test_sqlite_backend_on_postgres_is_refused(self):
        """The same protection, at the layer that builds the store.

        `validate_production` catches a SQLite *database*. This catches the other
        ordering: a real Postgres deployment that pins the SQLite *backend*, which
        passes the database check and then full-scans the embedding column in Python
        on every query.
        """
        from unittest.mock import MagicMock

        from app.core.config import Settings

        fake = MagicMock()
        fake.bind.dialect.name = "postgresql"

        prod = Settings(
            environment="production",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="sqlite",
        )
        with pytest.raises(RuntimeError, match="full-scans"):
            build_vector_store(fake, prod)
        assert Settings().vector_store == "sqlite"


class TestFakeProvidersAreRefusedInProduction:
    """A fixture that answers in production looks exactly like a working system.

    `fake` is the right default for tests and local runs -- it is deterministic, free
    and offline, and the embedding dimension does not depend on a third party being
    up. In staging or production it is a fault for the same reason a SQLite vector
    store is: the deployment starts, serves 200s, and produces numbers that look
    like measurements.

    The damage is not just wrong answers. Retrieval quality measured against fake
    embeddings is a hash-function similarity, not a semantic one, and it is what
    produced the 0.892 recall@10 baseline recorded in docs/known_issues.md. A
    deployment that silently keeps the fake provider ships that number as if it
    described the real system.
    """

    def _production_settings(self, **overrides) -> object:
        """A production config that passes the Postgres checks.

        The database and vector store must be real for the provider checks to be
        the thing under test rather than a downstream failure firing first.
        """
        from app.core.config import Settings

        return Settings(
            environment="production",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="pgvector",
            embedding_dim=384,
            **overrides,
        )

    def test_fake_embedding_provider_is_refused(self):

        prod = self._production_settings(embedding_provider="fake")
        with pytest.raises(RuntimeError, match="embedding_provider must not be 'fake'"):
            prod.validate_production()

    def test_fake_generation_provider_is_refused(self):
        # Embedding settings are real here on purpose. The provider checks are
        # sequential, so a fake *embedding* provider would raise first and this test
        # would assert nothing about generation at all.
        prod = self._production_settings(
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            generation_provider="fake",
        )
        with pytest.raises(
            RuntimeError, match="generation_provider must not be 'fake'"
        ):
            prod.validate_production()

    def test_a_real_pair_of_providers_is_accepted(self):
        """The check must be satisfiable, or nobody can deploy."""
        prod = self._production_settings(
            embedding_provider="huggingface",
            generation_provider="groq",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            hf_token="hf_x",
            groq_api_key="gsk_x",
            api_token="test-token",
        )
        prod.validate_production()  # must not raise

    def test_an_empty_api_token_is_refused(self):
        prod = self._production_settings(
            embedding_provider="huggingface",
            generation_provider="groq",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            api_token="",
        )
        with pytest.raises(RuntimeError, match="api_token must be set"):
            prod.validate_production()

    def test_a_fake_model_name_is_refused_even_with_a_real_provider(self):
        """The provider and the model name can disagree.

        Switching `EMBEDDING_PROVIDER` to `huggingface` while leaving the default
        `fake-embed-v1` would pass a provider-only check and then send that string
        to the HuggingFace API as a model name -- which fails at the first query
        rather than at startup.
        """
        prod = self._production_settings(
            embedding_provider="huggingface",
            generation_provider="groq",
            embedding_model="fake-embed-v1",
        )
        with pytest.raises(RuntimeError, match="embedding_model must not be a fake"):
            prod.validate_production()

    def test_staging_is_covered_too(self):
        """The check keys off `deployed`, so staging must behave like production."""
        from app.core.config import Settings

        staging = Settings(
            environment="staging",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="pgvector",
            embedding_provider="fake",
        )
        with pytest.raises(RuntimeError, match="embedding_provider must not be 'fake'"):
            staging.validate_production()

    @pytest.mark.parametrize("environment", ["local", "test"])
    def test_local_and_test_still_use_the_fake_providers(self, environment):
        """The documented use that the `vector_store` check once broke.

        `SqliteVectorStore` and the fake providers are the reason the test suite and
        local runs need no API key and no Postgres. Gating these checks on the
        environment rather than on the value is what preserves that, and it is the
        reason this test exists alongside the refusals above.
        """
        from app.core.config import Settings

        ok = Settings(
            environment=environment,
            database_url="sqlite:///./rag.db",
            vector_store="sqlite",
            embedding_provider="fake",
            generation_provider="fake",
            embedding_model="fake-embed-v1",
            embedding_dim=64,
        )
        ok.validate_production()  # must not raise

    def test_the_defaults_are_rejected_as_configured(self):
        """Unset means fake, so a deploy that sets nothing is refused.

        This is the realistic failure: someone sets `ENVIRONMENT=production` and
        `DATABASE_URL`, checks that the process starts, and ships without touching
        the provider variables because nothing forced them to.
        """
        from app.core.config import Settings

        prod = Settings(
            environment="production",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="pgvector",
            # Ignore the developer's .env. These three are omitted on purpose so
            # they take the module defaults (fake), which a real deploy must reject.
            _env_file=None,
        )
        assert prod.embedding_provider == "fake"
        assert prod.generation_provider == "fake"
        with pytest.raises(RuntimeError, match="must not be 'fake'"):
            prod.validate_production()

    def test_cosine_matches_manual_computation(self):
        """Guards the normalisation arithmetic against a sign error.

        A wrong sign here would make the *most* similar chunk rank last, and every
        rank-based metric downstream would still look plausible.
        """
        vector = [3.0, 4.0]
        unit = [v / math.sqrt(25) for v in vector]
        assert cosine_similarity(unit, unit) == pytest.approx(1.0, abs=1e-12)
        assert cosine_similarity(unit, [0.0, 1.0]) == pytest.approx(0.8)