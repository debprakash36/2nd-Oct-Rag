"""Vector store: the ANN/vector half of hybrid retrieval (FR-12).

Three implementations behind one protocol:

* `PgVectorStore` — pgvector on PostgreSQL. This is the production path and the
  only one that uses the HNSW index created in `alembic/versions/0001_init.py`.
* `ChromaVectorStore` — ChromaDB. Also a real ANN index, so it is a legitimate
  production option rather than a test double, and it can run alongside
  PostgreSQL. It answers with candidate ids only; text, ACL tags, and liveness are
  re-read from SQL before anything is returned (see the class docstring).
* `SqliteVectorStore` — exact brute-force cosine in Python over the JSON
  embedding column. This exists because the local and test environment has no
  PostgreSQL, and it is emphatically **not** the production store: it is a full
  scan, so it does not survive the 1M-chunk scale of NFR-4. It is here so
  retrieval behaviour is testable and the eval harness can run offline, not so
  the scale requirement can be waved away.

The `VectorStore` protocol is the seam required by architecture.md §7.3. Nothing
outside this module should know which implementation it holds.

**Filters are pushed into the query, never applied afterwards.** architecture.md
§3.4 is explicit that post-filtering is a security bug: text that reached the
model can leak by paraphrase even after the answer is scrubbed. Both
implementations therefore restrict the candidate scan to live documents before
scoring, and ACL enforcement happens before `limit` is applied so an
unauthorised chunk can never consume a slot in the result window.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session

from app.core.errors import StoreUnavailableError
from app.db.models import Chunk, Document, DocumentState
from app.retrieval.types import AccessFilter, RetrievalCandidate, Stage

log = logging.getLogger(__name__)

#: Extra candidates read beyond `k` to absorb ACL rejection before the limit.
#: Without headroom, a caller with narrow grants sees fewer than k results purely
#: because chunks it may not read filled the window.
_SCAN_HEADROOM = 50


def acl_sql_predicate(
    session: Session, access: AccessFilter | None, column: Any
) -> Any | None:
    """Dialect-safe ACL pre-filter for an array-like JSON column, or `None`.

    `JSON.contains()` is only valid on PostgreSQL, where it compiles to the JSONB
    containment operator. On SQLite it raises, and even where it does not it
    compares serialised JSON rather than the tag list — which is why the Python
    check in `_enforce` remains authoritative on every path.

    Returning `None` means "no SQL pre-filter available"; the caller must still
    enforce access in Python. The pre-filter is an optimisation for scale, never
    the correctness boundary.
    """
    if access is None or not access.allowed_tags:
        return None
    bind = session.get_bind()
    if bind is None or bind.dialect.name != "postgresql":
        return None
    from sqlalchemy.dialects.postgresql import JSONB

    jsonb_column = column.cast(JSONB)
    return or_(*(jsonb_column.contains(tag) for tag in access.allowed_tags))


@runtime_checkable
class VectorStore(Protocol):
    """Vector similarity search over chunk embeddings."""

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        """Return up to `k` chunks ordered by descending cosine similarity.

        Implementations must apply the access filter before returning, so that a
        caller never observes a candidate it is not permitted to read.
        """
        ...

    def count(self) -> int:
        """How many chunks this store can currently return.

        Exists so the configured backend can be compared against the authoritative
        chunk count. The two SQL-backed stores answer from the same table they
        count, so they always agree; only a derived index (Chroma) can differ, which
        is exactly the case worth surfacing. See `measure_divergence`.
        """
        ...


@dataclass(frozen=True)
class IndexDivergence:
    """How far the serving store is behind the authoritative chunks.

    Reported rather than inferred. The bug this exists to prevent is a working tree
    where the configured backend and the data on disk disagree and neither says so:
    a populated `data/chroma` beside a process serving from SQL reads as "the index
    is in use" and is not.
    """

    sql_chunks: int
    store_chunks: int

    @property
    def missing(self) -> int:
        return max(0, self.sql_chunks - self.store_chunks)

    @property
    def is_empty(self) -> bool:
        """The store cannot answer, but the corpus has something to answer with.

        The strict comparison on `sql_chunks` is deliberate. A fresh install with no
        documents at all is `0 == 0` and is *not* a fault -- treating it as one would
        make the health check return 503 on every new deployment and make an empty
        database uninstallable.
        """
        return self.store_chunks == 0 and self.sql_chunks > 0

    @property
    def is_stale(self) -> bool:
        return self.missing > 0

    def summary(self) -> str:
        return f"{self.store_chunks}/{self.sql_chunks} live chunks"


def live_chunk_count(session: Session) -> int:
    """Live chunks carrying an embedding: what any store must be able to serve.

    Mirrors the predicate in `_base_query` exactly -- `LIVE` documents only, and
    non-null embeddings only. If these two ever drift apart, a store can look empty
    here while `search` still returns candidates, and the divergence check would
    report a fault that does not exist.
    """
    return int(
        session.execute(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(
                Document.state == DocumentState.LIVE,
                Chunk.embedding.is_not(None),
            )
        ).scalar_one()
    )


def measure_divergence(session: Session, store: VectorStore) -> IndexDivergence:
    """Compare what the configured store can answer against what SQL holds.

    Polymorphic rather than an `isinstance` check: the SQL-backed stores return the
    authoritative count from their shared base implementation, and the derived store
    overrides it. There is no branch here to get wrong when a fourth backend appears.
    """
    sql_chunks = live_chunk_count(session)
    return IndexDivergence(sql_chunks=sql_chunks, store_chunks=store.count())


def assert_store_usable(session: Session, store: VectorStore) -> IndexDivergence:
    """Refuse to serve from a store that cannot answer; report one that lags.

    An empty derived index is not a degraded mode, it is a silent refusal to answer.
    `search` returns nothing, the pipeline abstains, and the user is told the corpus
    holds no relevant information when in fact nothing was ever searched. For a
    system whose whole claim is grounded answers that is the worst failure
    available, so it raises rather than degrading.

    A merely-stale index is different in kind. Ingestion writes SQL and never the
    index (architecture.md 4.3), so a derived index is behind by construction after
    any corpus change and the correct remedy is to re-project rather than to refuse
    to start. That is a warning, and it is also reported by `/health`.
    """
    divergence = measure_divergence(session, store)
    if divergence.is_empty:
        raise StoreUnavailableError(
            f"{type(store).__name__} holds 0 vectors but {divergence.sql_chunks} live "
            f"chunks exist in SQL. Every query would abstain as though the corpus had "
            f"no answer. Rebuild the index: python scripts/sync_chroma_index.py"
        )
    if divergence.is_stale:
        # Percent-style, not `extra=`: this module logs through stdlib
        # `logging`, not the project's structured logger, so `extra` keys would not
        # reach the JSON stream that trace_id and stage fields rely on.
        log.warning(
            "vector index is behind the authoritative chunks: %s (%d missing). "
            "Re-project with scripts/sync_chroma_index.py",
            divergence.summary(),
            divergence.missing,
        )
    return divergence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 for a zero-magnitude vector rather than raising or returning NaN.
    A zero vector has no direction, so its similarity to anything is genuinely
    undefined; 0.0 is the conservative answer because it ranks such a chunk last
    instead of first.
    """
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} != {len(right)}")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right, strict=True):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def _hydrate(row: Any) -> RetrievalCandidate:
    """Build a candidate from a (Chunk, Document.acl_tags) row tuple."""
    chunk: Chunk = row[0]
    acl_tags: list[str] = row[1] if len(row) > 1 else []
    return RetrievalCandidate(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        breadcrumb=chunk.breadcrumb,
        page=chunk.page,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        token_count=chunk.token_count,
        acl_tags=list(acl_tags or []),
    )


class _BaseVectorStore:
    """Shared query construction and access enforcement."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def count(self) -> int:
        """Count what `search` could return.

        Correct for the SQL-backed stores by construction: they read the same
        authoritative table this counts, so a derived index is the only backend
        where the answer can differ from `live_chunk_count`.
        """
        return live_chunk_count(self._session)

    def _base_query(self, access: AccessFilter | None) -> Select[Any]:
        """Select live chunks joined to their document for ACL tags.

        The `Document.state == LIVE` predicate is applied in SQL so tombstones
        (architecture.md §4.1) drop out of the candidate set before any scoring
        work happens, rather than being filtered out of an already-ordered result.
        """
        query = (
            select(Chunk, Document.acl_tags)
            .join(Document, Document.doc_id == Chunk.doc_id)
            .where(
                Document.state == DocumentState.LIVE,
                Chunk.embedding.is_not(None),
            )
        )
        acl = acl_sql_predicate(self._session, access, Document.acl_tags)
        if acl is not None:
            # Postgres only: keeps unauthorised rows out of the ANN index scan.
            # On other dialects the Python check in `_enforce` does the work.
            query = query.where(acl)
        # `.where()` re-derives the Select's type parameters from the criteria, so
        # the declared `Select[Any]` return type has to be asserted rather than
        # inferred. The selected columns are fixed by this method.
        return cast(Select[Any], query)

    def _enforce(
        self, candidates: list[RetrievalCandidate], access: AccessFilter | None
    ) -> list[RetrievalCandidate]:
        """Second ACL gate in Python, applied before `limit`.

        Belt-and-braces with the SQL predicate. `JSON.contains` compiles to `@>`
        on PostgreSQL and to a `LIKE` on SQLite, and the `LIKE` form can match a
        tag as a *substring* of another tag — which fails open. Checking the parsed
        Python list closes that hole, and doing it before the limit keeps
        unauthorised chunks from occupying result slots.
        """
        if access is None or not access.allowed_tags:
            return candidates
        return [c for c in candidates if access.permits(c.acl_tags)]

    def _log_query(self, k: int, access: AccessFilter | None, returned: int) -> None:
        log.debug(
            "vector search",
            extra={
                "k": k,
                "access": access.describe() if access else "unrestricted",
                "returned": returned,
            },
        )


class PgVectorStore(_BaseVectorStore):
    """pgvector-backed ANN search. The production vector store.

    Uses `<=>` (cosine distance) so the HNSW index created by the initial migration
    is usable. Cosine rather than L2 because embeddings are L2-normalised and
    cosine distance is the metric that matches how they were produced.
    """

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        # Over-fetch so ACL rejection still leaves k candidates. See
        # _SCAN_HEADROOM.
        limit = min(k + _SCAN_HEADROOM, k * 4)
        query = self._base_query(access)
        vector = list(query_vector)

        # `embedding.is_not(None)` is in the WHERE clause but mypy cannot narrow a
        # column expression through it, and a NULL here would raise inside
        # pgvector rather than skip the row.
        #
        # The `distance` callable is built from the *column*, not from each row,
        # because the ORM attribute is declared `list[float]` (the SQLite shape)
        # while the live value is a pgvector type. `.cos_distance` exists only on
        # the latter, so the static type and the runtime type genuinely differ
        # here and the call is made through an explicitly widened reference.
        distance = cast(Any, Chunk.embedding).cos_distance(vector)
        rows = (
            self._session.execute(query.order_by(distance).limit(limit))
            .all()
        )

        out: list[RetrievalCandidate] = []
        for row in rows:
            chunk: Chunk = row[0]
            stored = chunk.embedding
            if stored is None:
                continue
            candidate = _hydrate(row)
            # pgvector returns cosine *distance*; convert so that higher is better
            # and the scale is comparable with the SQLite path.
            candidate.scores[Stage.VECTOR] = 1.0 - float(
                cast(Any, stored).cos_distance(vector)
            )
            out.append(candidate)

        permitted = self._enforce(out, access)[:k]
        self._log_query(k, access, len(permitted))
        return permitted


class SqliteVectorStore(_BaseVectorStore):
    """Exact brute-force cosine search. Local development and tests only.

    Loads every live chunk with an embedding and scores in Python. Correctness is
    identical to the Postgres path; only the cost model differs, and the cost
    model is the entire reason production uses pgvector.
    """

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        rows = self._session.execute(self._base_query(access)).all()

        scored: list[tuple[float, RetrievalCandidate]] = []
        for row in rows:
            chunk: Chunk = row[0]
            embedding = chunk.embedding
            if embedding is None:
                # Excluded by the WHERE clause; re-checked because a JSON column
                # can hold null inside a non-null row shape on some dialects.
                continue
            candidate = _hydrate(row)
            # Fail loudly on a dimension mismatch (architecture.md §7.3): the
            # alternative is silently wrong similarity scores, which look like a
            # retrieval quality regression and get "fixed" by retuning thresholds.
            if len(embedding) != len(query_vector):
                raise ValueError(
                    f"embedding dimension mismatch for chunk {chunk.chunk_id}: "
                    f"stored {len(embedding)}, query {len(query_vector)}"
                )
            similarity = cosine_similarity(list(query_vector), list(embedding))
            candidate.scores[Stage.VECTOR] = similarity
            scored.append((similarity, candidate))

        # Sort by score descending, then a stable tiebreak. Without the tiebreak,
        # equal-similarity candidates come back in scan order and citation
        # rendering becomes non-deterministic between runs.
        scored.sort(key=lambda pair: (-pair[0], pair[1].doc_id, pair[1].chunk_index))

        permitted = self._enforce([c for _, c in scored], access)[:k]
        self._log_query(k, access, len(permitted))
        return permitted


#: Chunks per Chroma upsert during a sync. Bounded because Chroma serialises each
#: batch into its own segment write; one enormous batch is a single large lock.
_CHROMA_SYNC_BATCH = 256


def open_chroma_collection(settings: Any) -> Any:
    """Return the configured Chroma collection, creating it if absent.

    Imported lazily so `chromadb` is only required when the Chroma backend is
    actually selected. A hard module-level import would make the dependency
    mandatory for every deployment, including the pgvector production path that
    has no use for it.

    Created with `hnsw:space=cosine` to match the metric pgvector is queried with
    in `PgVectorStore`. The two stores must agree on the metric or their scores
    are not comparable and a threshold swept against one is meaningless against
    the other (FR-15).

    An existing collection is returned as-is rather than re-created: re-creating
    would silently drop every vector and turn a search into an empty result.
    """
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - depends on install profile
        raise RuntimeError(
            "vector_store is 'chroma' but chromadb is not installed. "
            "Install it with `pip install chromadb`."
        ) from exc

    client = chromadb.PersistentClient(path=settings.chroma_path)
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


class ChromaVectorStore(_BaseVectorStore):
    """ChromaDB-backed ANN search.

    **Chroma narrows the candidates; SQL decides what is allowed out of them.**
    That split is deliberate and is the whole reason this class is safe to add.

    A Chroma collection cannot enforce FR-7 or FR-16 on its own. Toggling a
    document to `disabled` writes nothing to the Chroma collection, and editing a
    document's `acl_tags` likewise lives only in SQL. A store that trusted its own
    metadata for either check would keep serving a tombstone and keep leaking a
    restricted chunk until the next full reindex — the exact "tombstone stops
    serving immediately" guarantee in `models.py:187` would silently become
    "at the next reindex". So every id Chroma returns is re-read through
    `_base_query`, which carries the `Document.state == LIVE` predicate and the
    ACL pre-filter, and is then checked in Python by `_enforce` before `limit`.

    The cost of that gate is one indexed `chunk_id IN (...)` lookup per query
    rather than a join, which is why it is affordable and why it is not merely a
    belt-and-braces check like it is in the SQL-backed stores: here it is the
    primary correctness boundary.

    This store also does not populate itself. Nothing in the ingest path writes to
    Chroma, so `Chunk` remains the sole writer of truth and `sync_chroma_index`
    is the explicit projection into it. Search against an unsynced collection
    returns nothing, which is why `sync_chroma_index` is part of this module rather
    than a script nobody finds.
    """

    def __init__(self, session: Session, collection: Any, *, dim: int) -> None:
        super().__init__(session)
        self._collection = collection
        self._dim = dim

    @classmethod
    def from_settings(cls, session: Session, settings: Any) -> ChromaVectorStore:
        """Build a store from settings. Used by `build_vector_store`."""
        return cls(
            session,
            open_chroma_collection(settings),
            dim=settings.embedding_dim,
        )

    def count(self) -> int:
        """Vectors actually present in the collection.

        Overrides the base count because this store holds a *copy*. Nothing in the
        ingest path writes here, so this number is only as current as the last
        `sync_chroma_index` run -- which is precisely the value `measure_divergence`
        exists to compare against SQL.
        """
        return int(self._collection.count())

    def search(
        self,
        query_vector: Sequence[float],
        *,
        k: int = 20,
        access: AccessFilter | None = None,
    ) -> list[RetrievalCandidate]:
        if len(query_vector) != self._dim:
            # Checked here rather than left to Chroma's own error so the message
            # names the pinned config value, which is the actionable part. The
            # existing test for this invariant asserts on "dimension mismatch".
            raise ValueError(
                f"embedding dimension mismatch: query has {len(query_vector)}, "
                f"the collection is built for {self._dim}. The embedding model has "
                "probably changed; the Chroma index must be rebuilt."
            )

        # A query against an empty collection is pointless work and, on some
        # Chroma versions, an error rather than an empty result.
        if self._collection.count() == 0:
            self._log_query(k, access, 0)
            return []

        # Over-fetch so ACL and tombstone rejection still leaves k candidates.
        # See _SCAN_HEADROOM. Clamped to the collection size because Chroma's
        # `n_results` above the population is at best redundant work.
        limit = min(min(k + _SCAN_HEADROOM, k * 4), self._collection.count())

        result = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=limit,
            include=["distances"],
        )
        ids, distances = _unwrap_chroma_query(result)

        if not ids:
            self._log_query(k, access, 0)
            return []

        # Authoritative gate: only these ids, only live documents, only rows that
        # still exist. A Chroma id with no matching chunk row (a purged document,
        # or a stale entry from a partial sync) simply fails to rejoin here.
        rows = (
            self._session.execute(self._base_query(access).where(Chunk.chunk_id.in_(ids)))
            .all()
        )
        hydrated = {row[0].chunk_id: _hydrate(row) for row in rows}

        out: list[RetrievalCandidate] = []
        for chunk_id, distance in zip(ids, distances, strict=True):
            candidate = hydrated.get(chunk_id)
            if candidate is None:
                continue
            # Cosine distance -> similarity, so higher is better and the scale
            # matches the other two stores. A drifted or non-cosine collection
            # would produce out-of-range scores rather than raising, which is why
            # the space is pinned at collection creation above.
            candidate.scores[Stage.VECTOR] = 1.0 - float(distance)
            out.append(candidate)

        # `out` is already in ascending-distance order from Chroma, so no re-sort.
        permitted = self._enforce(out, access)[:k]
        self._log_query(k, access, len(permitted))
        return permitted


def _unwrap_chroma_query(result: Any) -> tuple[list[str], list[float]]:
    """Flatten Chroma's per-query nested result into one id/distance pair list.

    Chroma shapes `query` results as a list per query embedding even for a
    single query, so `result["ids"][0]`. Guarded for the `[None]` case Chroma
    returns when a query matches nothing, which otherwise raises `TypeError` on
    iteration rather than returning an empty result.
    """
    ids = result.get("ids") or []
    distances = result.get("distances") or []
    if not ids or not ids[0]:
        return [], []
    if not distances or not distances[0]:
        # Would otherwise surface as an opaque ValueError from the strict zip in
        # the caller. Means `include=["distances"]` was dropped or the collection
        # is misconfigured, neither of which is the caller's problem to debug.
        raise RuntimeError(
            "chroma query returned ids without distances; cannot score candidates"
        )
    return list(ids[0]), list(distances[0])


def sync_chroma_index(
    session: Session,
    collection: Any,
    *,
    dim: int,
    batch_size: int = _CHROMA_SYNC_BATCH,
    doc_id: str | None = None,
) -> dict[str, int]:
    """Project authoritative live chunks into the Chroma collection.

    Upserts every chunk of a live document that has an embedding, then removes
    collection entries whose chunk no longer exists in SQL. Returns
    `{"upserted", "removed", "skipped"}`.

    **This is a rebuild, not a live write path.** Nothing in ingest calls it, so
    `Chunk` remains the only writer and the collection is a derived index that
    can be dropped and regenerated at any time without losing source data
    (architecture.md 4.3). The consequence is that the collection is stale until
    this runs: a document ingested a moment ago is not searchable yet.

    `doc_id` restricts the upsert to one document, for incremental catch-up. The
    removal sweep still considers the whole collection, because the cheaper
    failure to notice is a stale id left behind by a purged document — and
    `SqliteVectorStore.purge_chunks` has no Chroma counterpart to clean it.

    Chunks are only written for `live` documents. That keeps tombstoned text out
    of the index entirely, so a collection can be rebuilt from a database with
    disabled documents in it without leaking them.
    """
    query = (
        select(Chunk, Document.acl_tags)
        .join(Document, Document.doc_id == Chunk.doc_id)
        .where(
            Document.state == DocumentState.LIVE,
            Chunk.embedding.is_not(None),
        )
    )
    if doc_id is not None:
        query = query.where(Chunk.doc_id == doc_id)
    rows = session.execute(query).all()

    upserted = 0
    live_ids: set[str] = set()
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for row in batch:
            chunk: Chunk = row[0]
            acl_tags: list[str] = row[1] if len(row) > 1 else []
            embedding = chunk.embedding
            if embedding is None:
                continue
            if len(embedding) != dim:
                # Same fail-loudly rule as `SqliteVectorStore`: a wrong-width vector
                # scores as if it meant something, and the symptom shows up later
                # as a retrieval quality regression rather than a config error.
                raise ValueError(
                    f"embedding dimension mismatch for chunk {chunk.chunk_id}: "
                    f"stored {len(embedding)}, expected {dim}"
                )
            ids.append(chunk.chunk_id)
            live_ids.add(chunk.chunk_id)
            embeddings.append(list(embedding))
            documents.append(chunk.text)
            metadatas.append(
                {
                    "doc_id": chunk.doc_id,
                    "chunk_index": int(chunk.chunk_index),
                    "breadcrumb": chunk.breadcrumb,
                    "token_count": int(chunk.token_count),
                    # Chroma metadata accepts only scalars, so tags are joined.
                    # Nothing reads this back for enforcement -- ACL is decided in
                    # SQL -- it is here so an operator inspecting the collection can
                    # see why a chunk was restricted.
                    "acl_tags": ",".join(acl_tags or []),
                }
            )
        if ids:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            upserted += len(ids)

    # Remove ids the authoritative store no longer knows about. Without this a
    # purged document's vectors are still reachable via the ANN index; they are
    # harmless (hydration drops them, since the chunk row is gone) but they
    # permanently occupy ANN result slots and drag recall down forever.
    stale = [i for i in collection.get(include=[])["ids"] if i not in live_ids]
    if stale:
        collection.delete(ids=stale)

    # Upserts whose row had a NULL embedding are counted, not silently skipped.
    skipped = len(rows) - upserted
    log.info(
        "chroma index synced",
        extra={
            "upserted": upserted,
            "removed": len(stale),
            "skipped": skipped,
            "doc_id": doc_id or "all",
        },
    )
    return {"upserted": upserted, "removed": len(stale), "skipped": skipped}


def build_vector_store(session: Session, settings: Any | None = None) -> VectorStore:
    """Return the explicitly configured vector store.

    Selected here rather than by the caller so the decision lives in one place,
    mirroring `app/providers/embedding.py::get_embedding_provider`.

    **No inference from the database dialect.** An earlier version resolved an
    `"auto"` setting by inspecting `session.bind.dialect.name`. That guess is what
    allowed a populated `data/chroma` and the serving backend to disagree with
    neither reporting it, and the two are not interchangeable: a derived ANN index
    is stale by construction while the SQL table is not.

    An unrecognised name is an error rather than a fallback. A typo in configuration
    should stop the process, not quietly select a backend nobody asked for.

    The dialect guards below exist because the backend and the database are not
    independent choices. `pgvector` needs a Postgres to query, and `sqlite` against
    Postgres compiles to a full scan of the embedding column in Python -- the exact
    cost model `PgVectorStore` exists to avoid and the reason NFR-4 rules it out.
    Both mismatches are caught here rather than surfacing as a confusing driver error
    on the first user request.

    The old staging/production dialect block is gone: it only applied to the removed
    `"auto"` branch, and `Settings.validate_production` already rejects a SQLite
    `database_url` in those environments.
    """
    backend = (
        getattr(settings, "vector_store", "sqlite") if settings is not None else "sqlite"
    )

    dialect = session.bind.dialect.name if session.bind is not None else "unbound"

    if backend == "pgvector":
        if dialect != "postgresql":
            raise RuntimeError(
                "vector_store='pgvector' requires PostgreSQL, but DATABASE_URL is "
                f"{dialect!r}. Set VECTOR_STORE=sqlite for a local SQLite run, or "
                f"point DATABASE_URL at Postgres."
            )
        return PgVectorStore(session)

    if backend == "sqlite":
        if dialect == "postgresql":
            raise RuntimeError(
                "vector_store='sqlite' against PostgreSQL full-scans the embedding "
                "column in Python, which is the cost model NFR-4 rules out. Set "
                "VECTOR_STORE=pgvector."
            )
        return SqliteVectorStore(session)

    if backend == "chroma":
        return ChromaVectorStore.from_settings(session, settings)

    raise ValueError(
        f"unknown vector_store {backend!r}; expected 'pgvector', 'sqlite', or 'chroma'"
    )
