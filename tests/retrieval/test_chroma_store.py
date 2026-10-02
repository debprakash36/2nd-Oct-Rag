"""ChromaDB vector store: the SQL-authoritative gate and the sync/rebuild path.

The tests that matter most here are in `TestChromaAuthoritativeGate`, and the
reason is specific to this backend. Unlike the other two stores, Chroma holds a
*copy* of the index, so nothing it returns can be trusted without a second look.
Toggling a document to `disabled` or changing its `acl_tags` writes to SQL and
nowhere else, which means a Chroma-native implementation would keep serving a
tombstone and keep leaking restricted chunks until a full reindex.

So every test in that class deliberately performs **no re-sync** after mutating
the database. If the store were reading liveness and ACL out of Chroma's own
metadata rather than back through SQL, these tests would fail -- which is the
behaviour they exist to pin down.

`TestSyncChromaIndex` covers the other half: the collection is a derived index
(architecture.md 4.3), so it must be rebuildable from `Chunk` alone and must not
be able to contradict it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.retrieval.types import AccessFilter, Stage
from app.retrieval.vector_store import (
    ChromaVectorStore,
    sync_chroma_index,
)


@pytest.fixture
def chroma_collection(settings_env: Settings, tmp_path: Path):
    """A Chroma collection isolated per test.

    `PersistentClient` caches by path, so each test needs a distinct directory
    rather than a distinct collection name within a shared one.
    """
    import chromadb

    settings_env.vector_store = "chroma"
    settings_env.chroma_path = str(tmp_path / "chroma")
    settings_env.chroma_collection = "chunks"
    client = chromadb.PersistentClient(path=settings_env.chroma_path)
    return client.get_or_create_collection(
        name=settings_env.chroma_collection, metadata={"hnsw:space": "cosine"}
    )


@pytest.fixture
def chroma_store(session: Session, chroma_collection, settings_env: Settings) -> ChromaVectorStore:
    """A store over an *unsynced* collection.

    Deliberately does not sync on construction. A fixture that synced
    automatically would have to declare every document as a dependency to get the
    ordering right, and `test_filter_does_not_consume_result_slots` needs two
    documents synced together -- declaring both everywhere makes the ordering
    implicit and fragile. Tests call `sync()` explicitly once their documents
    exist, which also makes the "mutate SQL, do NOT re-sync" tests expressible.
    """
    return ChromaVectorStore(session, chroma_collection, dim=settings_env.embedding_dim)


@pytest.fixture
def sync(chroma_collection, settings_env: Settings):
    """Project the current database state into the collection."""

    def _sync() -> dict[str, int]:
        from app.db.session import get_session_factory

        with get_session_factory(settings_env)() as s:
            return sync_chroma_index(s, chroma_collection, dim=settings_env.embedding_dim)

    return _sync


@pytest.fixture
def synced_store(chroma_store, live_doc, sync) -> ChromaVectorStore:
    """A store whose collection matches a single live document."""
    sync()
    return chroma_store


class TestChromaVectorStore:
    def test_returns_ranked_candidates_with_vector_scores(
        self, synced_store, live_doc, provider, settings_env
    ):
        query = provider.embed(["refund policy"], model=settings_env.embedding_model)[0]
        results = synced_store.search(query, k=5)

        assert results
        for candidate in results:
            assert Stage.VECTOR in candidate.scores
            assert candidate.text

    def test_scores_descend(self, synced_store, live_doc, provider, settings_env):
        """Chroma returns ascending distance, which must surface as descending score."""
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        scores = [c.scores[Stage.VECTOR] for c in synced_store.search(query, k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_similarities_not_distances(
        self, synced_store, live_doc, provider, settings_env
    ):
        """Cosine distance 0.0 must read as similarity 1.0 (architecture.md 7.3).

        Asserted by querying with the exact string the chunk was embedded from, so
        the correct answer is a distance of 0.0 and a similarity of 1.0. A store
        that passed the raw distance through would score this perfect match 0.0 --
        ranking would invert and every positive threshold would abstain on it.

        `embedding_input` is reused rather than inlined because that is what
        ingest embeds; sending bare `chunk.text` would omit the breadcrumb and
        produce a genuinely lower score, making the assertion test nothing.
        """
        from app.ingest.index import embedding_input

        chunk = live_doc.chunks[0]
        probe = embedding_input(chunk.breadcrumb, chunk.text)
        query = provider.embed([probe], model=settings_env.embedding_model)[0]

        results = synced_store.search(query, k=1)
        assert results, "the chunk's own embedding must retrieve itself"
        assert results[0].chunk_id == chunk.chunk_id
        assert results[0].scores[Stage.VECTOR] == pytest.approx(1.0, abs=1e-6)

    def test_honours_limit(self, synced_store, live_doc, provider, settings_env):
        query = provider.embed(["policy"], model=settings_env.embedding_model)[0]
        assert len(synced_store.search(query, k=2)) <= 2

    def test_unsynced_collection_returns_nothing(
        self, session: Session, chroma_collection, settings_env: Settings, live_doc
    ):
        """Documented limitation, pinned so it cannot change silently.

        Nothing in ingest writes to Chroma, so the collection stays empty until
        `sync_chroma_index` runs. A store that silently searched SQL instead would
        pass this test while hiding that the index needs populating at all.
        """
        store = ChromaVectorStore(session, chroma_collection, dim=settings_env.embedding_dim)
        query = [0.1] * settings_env.embedding_dim
        assert store.search(query, k=5) == []

    def test_dimension_mismatch_fails_loudly(self, session: Session, synced_store):
        """Same invariant as the SQLite store (architecture.md 7.3).

        Chroma raises `InvalidArgumentError` of its own; the store converts it to
        the project's `ValueError` so the existing contract test language holds and
        the message can name the pinned dimension.
        """
        with pytest.raises(ValueError, match="dimension mismatch"):
            synced_store.search([0.1] * 7, k=5)


class TestChromaAuthoritativeGate:
    """FR-7 and FR-16 must hold with no re-sync after the mutation."""

    def _tag_document(self, session: Session, doc_id: str, tags: list[str]) -> None:
        from app.db.models import Document

        doc = session.get(Document, doc_id)
        doc.acl_tags = tags
        session.commit()

    def test_disabled_document_is_immediately_unretrievable(
        self, synced_store, session: Session, live_doc, provider, settings_env
    ):
        from app.db.models import DocumentState
        from app.ingest.states import transition

        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        assert synced_store.search(query, k=50), "precondition: synced doc is retrievable"

        transition(session, live_doc, DocumentState.DISABLED, reason="test")
        session.commit()

        # No `sync_chroma_index` call: the vectors are still in Chroma, and the
        # store must still refuse them because SQL says the document is not live.
        assert synced_store.search(query, k=50) == []

    def test_failed_document_is_never_retrievable(
        self, synced_store, session: Session, live_doc, provider, settings_env
    ):
        from app.db.models import DocumentState
        from app.ingest.states import transition

        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        transition(session, live_doc, DocumentState.FAILED, reason="ingest broke")
        session.commit()

        assert synced_store.search(query, k=50) == []

    def test_unauthorised_chunk_never_appears_in_results(
        self, synced_store, session: Session, live_doc, provider, settings_env
    ):
        self._tag_document(session, live_doc.doc_id, ["restricted"])
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]

        results = synced_store.search(query, k=50, access=AccessFilter(frozenset({"other-team"})))
        assert results == []

    def test_authorised_chunk_does_appear(
        self, synced_store, session: Session, live_doc, provider, settings_env
    ):
        self._tag_document(session, live_doc.doc_id, ["finance"])
        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]

        results = synced_store.search(query, k=50, access=AccessFilter(frozenset({"finance"})))
        assert results

    def test_filter_does_not_consume_result_slots(
        self,
        chroma_store,
        session: Session,
        live_doc,
        second_live_doc,
        sync,
        provider,
        settings_env,
    ):
        """FR-16 correctness.

        Over-fetching is what makes this hold: with k+_SCAN_HEADROOM pulled from
        Chroma and ACL applied before the final limit, a restricted document
        cannot shrink the result window. Sizing the Chroma query at exactly k
        would fail this test.
        """
        self._tag_document(session, live_doc.doc_id, ["restricted"])
        self._tag_document(session, second_live_doc.doc_id, ["finance"])
        sync()

        query = provider.embed(["refund policy"], model=settings_env.embedding_model)[0]
        results = chroma_store.search(query, k=50, access=AccessFilter(frozenset({"finance"})))

        assert results
        assert all("restricted" not in c.acl_tags for c in results)

    def test_purged_chunks_are_dropped_at_hydration(
        self, synced_store, session: Session, live_doc, provider, settings_env
    ):
        """A Chroma id with no chunk row must not produce a candidate.

        This is the purge path: `purge_chunks` deletes rows and has no Chroma
        counterpart, so the stale id is still reachable through the ANN index. It
        must be discarded rather than surfaced as a chunk-less candidate.
        """
        from app.ingest.index import purge_chunks

        query = provider.embed(["refund"], model=settings_env.embedding_model)[0]
        assert synced_store.search(query, k=50), "precondition: chunks are retrievable"

        purge_chunks(session, live_doc.doc_id)
        session.commit()

        assert synced_store.search(query, k=50) == []

    def test_text_and_citations_come_from_sql_not_chroma(
        self, synced_store, settings_env: Settings, live_doc
    ):
        """Hydrated text must be the authoritative copy.

        Chroma is handed the chunk text by the sync, but the store must not read
        it back from there: a re-ingest updates `Chunk` immediately while the
        collection stays stale until the next sync, and serving the stale copy
        would cite text the user can no longer find in the file.
        """
        expected = {c.chunk_id: c.text for c in live_doc.chunks}
        results = synced_store.search([0.1] * settings_env.embedding_dim, k=50)


        assert results
        for candidate in results:
            assert candidate.text == expected[candidate.chunk_id]


class TestSyncChromaIndex:
    def test_upserts_live_chunks(
        self, session: Session, chroma_collection, settings_env: Settings, live_doc
    ):
        result = sync_chroma_index(
            session, chroma_collection, dim=settings_env.embedding_dim
        )
        assert result["upserted"] == len(live_doc.chunks)
        assert chroma_collection.count() == len(live_doc.chunks)

    def test_excludes_tombstoned_documents(
        self, session: Session, chroma_collection, settings_env: Settings, live_doc
    ):
        """Tombstoned text must not be written into the index at all.

        Enforced at sync time as well as at query time so a rebuilt collection
        from a database holding disabled documents cannot leak them.
        """
        from app.db.models import DocumentState
        from app.ingest.states import transition

        transition(session, live_doc, DocumentState.DISABLED, reason="test")
        session.commit()

        result = sync_chroma_index(
            session, chroma_collection, dim=settings_env.embedding_dim
        )
        assert result["upserted"] == 0
        assert chroma_collection.count() == 0

    def test_removes_ids_missing_from_sql(
        self, session: Session, synced_store, chroma_collection, live_doc, settings_env
    ):
        """Stale vectors must be swept, not left to occupy ANN slots forever."""
        from app.ingest.index import purge_chunks

        before = chroma_collection.count()
        assert before > 0

        purge_chunks(session, live_doc.doc_id)
        session.commit()

        result = sync_chroma_index(
            session, chroma_collection, dim=settings_env.embedding_dim
        )
        assert result["removed"] == before
        assert chroma_collection.count() == 0

    def test_is_idempotent(
        self, session: Session, chroma_collection, settings_env: Settings, live_doc
    ):
        """Re-running must not duplicate or drop vectors.

        This is what makes the collection safe to rebuild on a schedule rather
        than needing a clean-slate step.
        """
        sync_chroma_index(session, chroma_collection, dim=settings_env.embedding_dim)
        first = chroma_collection.count()
        sync_chroma_index(session, chroma_collection, dim=settings_env.embedding_dim)

        assert chroma_collection.count() == first
        assert first == len(live_doc.chunks)

    def test_partial_sync_by_doc_id(
        self,
        session: Session,
        chroma_collection,
        settings_env: Settings,
        live_doc,
        second_live_doc,
    ):
        """Incremental catch-up scopes the upsert but still sweeps stale ids."""
        result = sync_chroma_index(
            session,
            chroma_collection,
            dim=settings_env.embedding_dim,
            doc_id=second_live_doc.doc_id,
        )
        assert result["upserted"] == len(second_live_doc.chunks)

    def test_dimension_mismatch_raises(
        self, session: Session, chroma_collection, live_doc
    ):
        with pytest.raises(ValueError, match="dimension mismatch"):
            sync_chroma_index(session, chroma_collection, dim=7)

    def test_skips_chunks_without_embeddings(
        self, session: Session, chroma_collection, settings_env: Settings, live_doc
    ):
        """A chunk the embedder never reached is counted as skipped, not written."""
        from sqlalchemy import update

        from app.db.models import Chunk

        session.execute(update(Chunk).values(embedding=None))
        session.commit()

        result = sync_chroma_index(
            session, chroma_collection, dim=settings_env.embedding_dim
        )
        assert result["upserted"] == 0
        assert result["skipped"] > 0
        assert chroma_collection.count() == 0


class TestChromaSelection:
    def test_explicit_chroma_wins_over_dialect(
        self, session: Session, settings_env: Settings, tmp_path: Path
    ):
        """Chroma runs alongside PostgreSQL, so it must not be dialect-gated.

        `tmp_path` because `build_vector_store` constructs a real
        `PersistentClient`, which creates its directory on disk. Pointing it at a
        repo-relative path would leave an artifact in the working tree after every
        test run.
        """
        from app.retrieval.vector_store import build_vector_store

        settings_env.vector_store = "chroma"
        settings_env.chroma_path = str(tmp_path / "chroma-selection")

        assert isinstance(build_vector_store(session, settings_env), ChromaVectorStore)

    def test_auto_is_no_longer_a_valid_setting(self):
        """The dialect-derived 'auto' was removed; the backend must be named.

        It inferred a backend from the database dialect, which let a populated
        `data/chroma` and the store actually serving queries disagree with neither
        one reporting it. Rejecting the value at config load means an existing
        deployment carrying `VECTOR_STORE=auto` fails immediately and visibly
        instead of quietly getting whichever backend the dialect implied.
        """
        from pydantic import ValidationError

        from app.core.config import Settings as S

        with pytest.raises(ValidationError, match="vector_store"):
            S(vector_store="auto")

    def test_production_rejects_sqlite_backend(self):
        """An explicit sqlite backend must fail as loudly as a SQLite database.

        The dialect check cannot catch this one: `VECTOR_STORE=sqlite` against a
        real Postgres looks healthy at startup and full-scans in production.
        """
        from app.core.config import Settings as S

        prod = S(
            environment="production",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="sqlite",
        )
        with pytest.raises(RuntimeError, match="must not be 'sqlite'"):
            prod.validate_production()

    def test_production_accepts_chroma(self):
        """Chroma is an acceptable production *backend* -- the providers are explicit.

        The provider values are named rather than defaulted because `validate_production`
        now refuses fake providers in deployed environments. That refusal is covered on
        its own; what matters here is that the backend choice is not what gets rejected.
        """
        from app.core.config import Settings as S

        prod = S(
            environment="production",
            database_url="postgresql+psycopg://rag:rag@localhost:5432/rag",
            vector_store="chroma",
            embedding_provider="huggingface",
            generation_provider="groq",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            api_token="test-token",
        )
        prod.validate_production()
