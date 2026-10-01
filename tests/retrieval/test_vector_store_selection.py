"""Vector store selection must be explicit, and must be checked against the data.

The defect this file exists for is narrower than "the wrong backend was chosen". The
`auto` setting inferred a backend from the database dialect, and that inference was
usually *correct* -- SQL held more chunks than the derived index did. The real failure
was that nothing compared the configured backend against the data that actually
existed, so a populated `data/chroma` sitting beside the process read as "the index
is in use" when it was not, with no signal from anywhere.

So two properties are pinned here:

1. A default-config run serves from the store that holds the indexed chunks, and can
   demonstrate that it does.
2. A derived index that is empty or behind is reported rather than absorbed.

Everything is seeded into the test database or a `tmp_path` collection. Nothing here
reads the checked-in `rag.db` or `data/chroma`; that comparison is the opt-in
`indexcheck` test in `test_checked_in_index.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import StoreUnavailableError
from app.retrieval.vector_store import (
    ChromaVectorStore,
    IndexDivergence,
    PgVectorStore,
    SqliteVectorStore,
    assert_store_usable,
    build_vector_store,
    live_chunk_count,
    measure_divergence,
)


def _embed(seed: int, dim: int = 384) -> list[float]:
    """A distinct, valid-width vector. Real embeddings come from the provider."""
    vector = [0.0] * dim
    vector[seed % dim] = 1.0
    return vector


def _seed_live_chunks(session: Session, n: int) -> list[str]:
    """Write `n` live, embedded chunks straight to `Chunk`.

    Bypasses ingestion deliberately: these tests are about what the selection and
    divergence logic can see, and an ingest path that itself depends on the
    configured store would make them circular.
    """
    from app.db.models import Chunk, Document, DocumentState

    doc = Document(
        doc_id="d1",
        filename="policy.md",
        mime_type="text/markdown",
        state=DocumentState.LIVE,
        content_hash="h1",
    )
    session.add(doc)
    ids = []
    for i in range(n):
        chunk_id = f"c{i:03d}"
        session.add(
            Chunk(
                chunk_id=chunk_id,
                doc_id="d1",
                chunk_index=i,
                text=f"chunk {i}",
                embedding=_embed(i),
                token_count=3,
                # `char_start`/`char_end` are NOT NULL; values are irrelevant here
                # because nothing in the selection or divergence path reads them.
                char_start=i * 10,
                char_end=(i + 1) * 10,
            )
        )
        ids.append(chunk_id)
    session.commit()
    return ids


class TestBackendIsExplicit:
    def test_auto_is_rejected_at_config_load(self):
        """A deployment still carrying `VECTOR_STORE=auto` must fail visibly.

        Silently keeping the value would be worse than removing it: the setting would
        look configured while doing nothing.
        """
        with pytest.raises(ValidationError, match="vector_store"):
            Settings(vector_store="auto")  # type: ignore[arg-type]

    def test_default_is_sqlite(self):
        """The default must be the authoritative store, not a guess.

        Chroma is a derived copy and is stale by construction after any ingest, so
        defaulting to it would make the default depend on whether someone remembered
        to re-project.
        """
        assert Settings().vector_store == "sqlite"

    def test_unknown_backend_is_an_error_not_a_fallback(
        self, session: Session, settings_env: Settings
    ):
        """A typo must stop the process, not select a backend nobody asked for."""
        settings_env.vector_store = "chromaa"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="unknown vector_store"):
            build_vector_store(session, settings_env)

    def test_pgvector_requires_postgres(self, session: Session, settings_env: Settings):
        """The backend and the database are not independent choices."""
        settings_env.vector_store = "pgvector"
        with pytest.raises(RuntimeError, match="requires PostgreSQL"):
            build_vector_store(session, settings_env)

    def test_sqlite_against_postgres_is_refused(self, settings_env: Settings):
        """`sqlite` on Postgres compiles to a full scan in Python (NFR-4)."""
        from unittest.mock import MagicMock

        settings_env.vector_store = "sqlite"
        fake = MagicMock()
        fake.bind.dialect.name = "postgresql"
        with pytest.raises(RuntimeError, match="full-scans"):
            build_vector_store(fake, settings_env)

    def test_explicit_pgvector_on_postgres_is_accepted(self, settings_env: Settings):
        from unittest.mock import MagicMock

        settings_env.vector_store = "pgvector"
        fake = MagicMock()
        fake.bind.dialect.name = "postgresql"
        assert isinstance(build_vector_store(fake, settings_env), PgVectorStore)


class TestDefaultConfigUsesTheStoreHoldingTheData:
    def test_default_run_serves_from_the_store_that_has_the_chunks(
        self, session: Session, settings_env: Settings
    ):
        """The regression this file exists for.

        A default-config run must serve from the store that actually holds the
        indexed chunks -- and be able to say so, rather than leaving an operator to
        notice a populated directory that nothing is reading.
        """
        _seed_live_chunks(session, n=12)

        store = build_vector_store(session, settings_env)

        # The right type is not sufficient: it must be the store that can answer.
        assert isinstance(store, SqliteVectorStore)
        divergence = measure_divergence(session, store)
        assert divergence.missing == 0
        assert divergence.sql_chunks == 12
        assert store.search(_embed(0), k=5), "the default store returned no candidates"

    def test_no_divergence_is_reported_for_an_sql_backed_store(
        self, session: Session, settings_env: Settings
    ):
        """SQL-backed stores answer from the table they count, so they cannot lag."""
        _seed_live_chunks(session, n=5)
        store = build_vector_store(session, settings_env)
        assert measure_divergence(session, store).is_stale is False
        assert store.count() == live_chunk_count(session) == 5

    def test_non_live_chunks_are_excluded_from_the_count(
        self, session: Session, settings_env: Settings
    ):
        """`live_chunk_count` must mirror the search predicate, not count tombstones.

        If it counted more than `search` could return, an index that is perfectly in
        sync would be reported as permanently behind.
        """
        from app.db.models import Document, DocumentState

        _seed_live_chunks(session, n=4)
        session.get(Document, "d1").state = DocumentState.DISABLED
        session.commit()

        store = build_vector_store(session, settings_env)
        assert live_chunk_count(session) == 0
        assert measure_divergence(session, store).missing == 0
        assert measure_divergence(session, store).is_empty is False


class TestDivergenceIsSurfaced:
    def test_empty_derived_index_raises_instead_of_abstaining(
        self, session: Session, settings_env: Settings, tmp_path
    ):
        """The worst failure mode: silent refusal to answer.

        An empty derived store returns no candidates, the pipeline abstains, and the
        user is told the corpus has no relevant information when nothing was ever
        searched. It must raise, and the message must name the remedy.
        """
        _seed_live_chunks(session, n=6)
        settings_env.vector_store = "chroma"
        settings_env.chroma_path = str(tmp_path / "chroma-empty")

        store = build_vector_store(session, settings_env)
        with pytest.raises(StoreUnavailableError, match="sync_chroma_index"):
            assert_store_usable(session, store)

    def test_stale_derived_index_warns_but_still_serves(
        self, session: Session, settings_env: Settings, tmp_path, caplog
    ):
        """Behind is not the same as empty.

        Ingestion never writes the index, so a derived index is behind by
        construction after any corpus change. Blocking would make `chroma` unusable
        in normal operation, so this warns -- but it must still say so.
        """
        ids = _seed_live_chunks(session, n=6)
        settings_env.vector_store = "chroma"
        settings_env.chroma_path = str(tmp_path / "chroma-stale")

        store = build_vector_store(session, settings_env)
        collection = store._collection

        # Project all but one, mimicking an ingest that landed after the last sync.
        from app.db.models import Chunk

        held_back = ids[-1]
        for cid in ids[:-1]:
            chunk = session.get(Chunk, cid)
            collection.upsert(
                ids=[cid],
                embeddings=[list(chunk.embedding)],
                documents=[chunk.text],
                metadatas=[{"doc_id": "d1", "chunk_index": 0}],
            )

        with caplog.at_level("WARNING"):
            divergence = assert_store_usable(session, store)

        assert divergence.store_chunks == 5
        assert divergence.missing == 1
        assert divergence.is_stale is True
        assert divergence.is_empty is False
        assert "behind" in caplog.text.lower()
        assert held_back not in collection.get(include=[])["ids"]

    def test_a_fresh_install_with_no_documents_is_not_a_fault(
        self, session: Session, settings_env: Settings
    ):
        """0 == 0 must not be treated as broken.

        Otherwise the new health check 503s every new deployment and an empty
        database becomes impossible to start against.
        """
        divergence = IndexDivergence(sql_chunks=0, store_chunks=0)
        assert divergence.is_empty is False
        assert divergence.is_stale is False
        assert divergence.missing == 0

        store = build_vector_store(session, settings_env)
        assert assert_store_usable(session, store) == divergence

    def test_summary_is_readable_in_a_health_response(self):
        assert IndexDivergence(sql_chunks=304, store_chunks=297).summary() == (
            "297/304 live chunks"
        )


class TestChromaCountTracksTheCollection:
    def test_count_reflects_the_collection_not_sql(
        self, session: Session, settings_env: Settings, tmp_path
    ):
        """`ChromaVectorStore.count` must read the collection.

        If it fell through to the base implementation it would return the SQL count
        and the divergence check would report perfect agreement forever -- which is
        the exact bug being fixed, hidden inside the fix.
        """
        _seed_live_chunks(session, n=3)
        settings_env.vector_store = "chroma"
        settings_env.chroma_path = str(tmp_path / "chroma-count")

        store = build_vector_store(session, settings_env)
        assert isinstance(store, ChromaVectorStore)
        assert store.count() == 0
        assert live_chunk_count(session) == 3
        assert measure_divergence(session, store).missing == 3
