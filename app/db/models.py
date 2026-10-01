"""ORM models for Phase 1: documents and chunks.

Design notes that are load-bearing rather than incidental:

* The chunk store is the **authoritative** membership record (architecture.md
  4.3). The vector index and keyword index are derived from it. Everything that
  decides what exists reads `Chunk`, never the derived indexes, so index drift
  is recoverable by rebuild instead of being a permanent correctness bug.

* `Document.state` gates retrieval. Only `live` is retrievable, enforced here at
  the query level rather than trusted to callers.

* `embedding` is stored on the chunk so the vector index lives in the same row
  as the text it describes. With pgvector this column is `vector(N)`; SQLite has
  no such type, so it falls back to a JSON blob. The search interface is provided
  by `VectorStore` in Phase 2, which is why the storage difference is contained
  to this module.
"""

from __future__ import annotations

import datetime as dt
import enum
import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql.type_api import TypeEngine


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _embedding_type(dim: int | None = None) -> TypeEngine[Any]:
    """Column type for an embedding: `vector(N)` on Postgres, JSON on SQLite.

    Resolved lazily rather than at import so importing this module does not
    require the dimension to be configured, and so tests can construct
    `Chunk(...)` under a different dimension without reimporting.

    The dimension comes from `Settings` and is pinned there. If it is ever
    changed, the existing column is the wrong width and every stored vector is
    invalid — the schema must be migrated, not merely re-read. The migration
    (alembic/versions/0001_init.py) bakes the same value into `vector(N)`.
    """
    if dim is None:
        from app.core.config import get_settings

        dim = get_settings().embedding_dim

    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:  # pragma: no cover - pgvector is a declared dependency
        return JSON()

    # `Vector` on the postgresql dialect, JSON everywhere else. `with_variant`
    # rather than a dialect check so `Base.metadata.create_all` and the migration
    # both see the same declaration.
    return JSON().with_variant(Vector(dim), "postgresql")


class DocumentState(enum.StrEnum):
    """Lifecycle states (architecture.md 4.1).

    The forward pipeline is linear. `superseded`, `disabled` and `deleted` are
    tombstones: the rows and their chunks remain, but retrieval filters on state
    so the document stops being served immediately. Physical removal is a
    separate background operation.

    `duplicate` is not in the architecture.md 4.1 state diagram, which predates it.
    It was added because "flag as near-duplicate" was implemented as a column and
    nothing more: a duplicate was still embedded, indexed and set `LIVE`, so eleven
    identical uploads all answered the same query. Flagging is not neutralising, and
    retrieval only knows about `LIVE`. It is a tombstone in the same sense as the
    others -- the row and its `duplicate_of` pointer are kept so an admin can promote
    it deliberately, which is the "offer to skip" half of FR-4.
    """

    PENDING = "pending"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    LIVE = "live"
    FAILED = "failed"
    SUPERSEDED = "superseded"
    DISABLED = "disabled"
    DELETED = "deleted"
    DUPLICATE = "duplicate"


class Document(Base):
    """An uploaded file and its ingestion state."""

    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)

    # Hash of the *chunked content*, not the file bytes (architecture.md 4.4).
    # Re-saving a PDF changes the bytes without changing the text; byte hashing
    # would re-embed for nothing.
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)

    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    state: Mapped[DocumentState] = mapped_column(
        Enum(DocumentState, native_enum=False, length=16),
        nullable=False,
        default=DocumentState.PENDING,
        index=True,
    )

    byte_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    uploaded_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    indexed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Pushed into retrieval filters (FR-16). Stored as JSON because the
    # authoritative store is Postgres, where a text[] column would be natural;
    # JSON keeps one definition working on SQLite for tests.
    acl_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Canonical cleaned text. Offsets on every chunk are relative to exactly
    # this string, which is what makes a citation resolve to an exact passage
    # instead of a whole file (architecture.md 4.2).
    cleaned_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Set when the upload matches an existing document's chunked content
    # (FR-4). Recorded rather than acted on silently, so an admin can decide.
    duplicate_of: Mapped[str | None] = mapped_column(String(32), nullable=True)

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )


class ChunkTerm(Base):
    """Keyword index row: one (chunk, term) frequency pair.

    Derived from `Chunk` and rebuildable from it (architecture.md 4.3). The chunk
    store stays the authoritative membership record, so dropping this table loses
    no source data — only query speed until it is rebuilt.

    `state` is denormalised from `Document`. Without it, every keyword query
    would join back to `documents` to enforce "only live is retrievable"; with
    it, the filter applies inside the term index scan, so a tombstone stops
    serving matches immediately rather than at the next reindex (FR-7).
    """

    __tablename__ = "chunk_terms"
    __table_args__ = (
        Index("ix_chunk_terms_term_state", "term", "state"),
        Index("ix_chunk_terms_chunk", "chunk_id"),
    )

    chunk_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chunks.chunk_id", ondelete="CASCADE"), primary_key=True
    )
    term: Mapped[str] = mapped_column(String(64), primary_key=True)
    term_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    state: Mapped[DocumentState] = mapped_column(
        Enum(DocumentState, native_enum=False, length=16), nullable=False
    )


class Chunk(Base):
    """A retrievable passage. Authoritative record of what exists."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
        Index("ix_chunk_doc", "doc_id"),
    )

    chunk_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"),
                                        nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Raw chunk text, WITHOUT the breadcrumb. The breadcrumb is prepended only
    # to the embedding input; storing it here would put document titles into
    # search results and into every citation.
    text: Mapped[str] = mapped_column(Text, nullable=False)
    breadcrumb: Mapped[str] = mapped_column(Text, nullable=False, default="")
    section_path: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Offsets into Document.cleaned_text. The invariant "cleaned_text[start:end]
    # reconstructs the chunk" is asserted by tests/ingest/test_offsets.py.
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)

    page: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Embedding lives beside the text it describes so the two cannot drift.
    #
    # The column is `vector(N)` on Postgres and a JSON blob on SQLite. Declared
    # as a variant rather than as plain `JSON` because the two are not
    # interchangeable at the type level: a JSON-typed attribute writing to a
    # real `vector` column relies on pgvector accepting JSON's serialisation as
    # a side effect of both formats being `[...]`. That works today and breaks
    # silently the day either format changes — a float that cannot round-trip is
    # a corrupted vector, and corruption is invisible until search quality
    # degrades.
    embedding: Mapped[list[float] | None] = mapped_column(_embedding_type(), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    document: Mapped[Document] = relationship(back_populates="chunks")


class Conversation(Base):
    """A chat thread (FR-23, architecture.md 5).

    Deliberately carries no user or tenant column. v1 has no multi-tenancy
    (architecture.md NG5) and FR-26 forbids cross-session memory of the user, so a
    conversation is reachable by id alone. Adding an owner later is a migration plus
    an access predicate on every read; pretending the column exists now would mean
    writing a predicate that checks nothing and looks like access control.
    """

    __tablename__ = "conversations"
    __table_args__ = (Index("ix_conversations_created", "created_at"),)

    conversation_id: Mapped[str] = mapped_column(
        String(32), primary_key=True, default=_new_uuid
    )
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    turns: Mapped[list[Turn]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        # `selectin` because every conversation read in Phase 4 is a list of
        # conversations shown with their turns; a lazy load per row would be N+1 on
        # exactly the endpoint the UI hits first.
        lazy="selectin",
        order_by="Turn.turn_index",
    )


class TurnRole(enum.StrEnum):
    """Who produced a turn."""

    USER = "user"
    ASSISTANT = "assistant"


class Turn(Base):
    """One message in a conversation (FR-23, architecture.md 5).

    `turn_index` is a monotonic counter rather than relying on `turn_id` ordering:
    the id is a random uuid, so ordering by it would return a conversation
    shuffled differently per read, and the last-N-turns context window would then
    contain an arbitrary slice of the thread.

    `citations` holds the chunk ids the answer actually cited, which is not the
    same as the retrieved set in `QueryLog.retrieved_ids` — a source panel shows
    everything read, a turn records everything claimed. The difference is what
    makes "the model cited this passage" answerable after the fact.
    """

    __tablename__ = "turns"
    __table_args__ = (
        Index("ix_turns_conversation", "conversation_id"),
        UniqueConstraint("conversation_id", "turn_index", name="uq_turn_conversation_index"),
    )

    turn_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[TurnRole] = mapped_column(
        Enum(TurnRole, native_enum=False, length=16), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    #: Chunk ids cited by an assistant turn. Empty for user turns.
    citations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    #: Links the turn back to the QueryLog row that produced it. Nullable and not a
    #: foreign key: a QueryLog row is written from inside the streaming generator,
    #: so the turn can be persisted before the log exists. FR-30 correlates the two.
    query_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: True when this turn is a refusal. Persisted so the UI can style a declined
    #: answer differently and so refusal rate can be recomputed from turns alone.
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    conversation: Mapped[Conversation] = relationship(back_populates="turns")


class QueryLog(Base):
    """One answered (or refused) query, with the fields FR-28 requires.

    This table is the substrate for FR-30, the PRD's improvement loop, and the
    architecture.md 7.2 dashboards. Every field is persisted from the first
    version because retroactive instrumentation is not possible: a query that was
    not logged with per-stage scores cannot be reclassified later as a content gap
    versus a retrieval gap (architecture.md 5). `scores` and `rewritten_query` are
    the two most often omitted and the two that make that classification
    possible, so they are non-optional here.

    `conversation_id` is a correlation key rather than a foreign key (see the
    column comment). `feedback` is a small nullable string (`up`/`down`) rather than
    an enum so the Phase 4 feedback endpoint can populate it without a schema
    migration.
    """

    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_trace", "trace_id"),
        Index("ix_query_logs_conversation", "conversation_id"),
        Index("ix_query_logs_created", "created_at"),
    )

    query_id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_new_uuid)
    #: End-to-end request id, shared with the structured logs (NFR-8).
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: Conversation this query belongs to. A bare string rather than a foreign key
    #: even though `conversations` now exists: a turn is persisted from inside the
    #: streaming generator and the QueryLog row is written in that same flush, so a
    #: hard foreign key would couple their ordering. Nothing joins on it for
    #: correctness — it is a correlation key (architecture.md 5, FR-30).
    conversation_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    #: The query as the user asked it. Redaction (FR-33) happens before this is
    #: written; the column already stores the post-redaction value.
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    #: The rewritten standalone query, logged separately (architecture.md 3.2).
    #: A bad rewrite is otherwise indistinguishable from a retrieval failure.
    rewritten_query: Mapped[str] = mapped_column(Text, nullable=False, default="")

    #: The chunk ids that were put in front of the model, in order.
    retrieved_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    #: Per-stage scores keyed by chunk id: {chunk_id: {vector, keyword, fused,
    #: reranked}}. The architecture calls the keyword stage "bm25"; the pipeline's
    #: stage name is `keyword`, and the value here is the pipeline's name.
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    threshold_applied: Mapped[float | None] = mapped_column(Float, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    #: Token accounting, approximate when the provider does not report usage.
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Time to the first user-visible token (NFR-1). Null only for a refusal that
    #: never called the model and emitted no token.
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    #: Fabricated markers removed. A rising count is the earliest degradation
    #: signal (architecture.md 7.2), which is why it is persisted, not just logged.
    citations_stripped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: `up` | `down` | null. Populated by the Phase 4 feedback endpoint (FR-29).
    feedback: Mapped[str | None] = mapped_column(String(8), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


def retrievable_chunk_filter() -> Any:
    """SQL predicate restricting rows to chunks of live documents.

    Phase 2 retrieval must use this rather than filtering in Python, both to
    push the predicate into the index and to keep the "only live is
    retrievable" invariant in one place.
    """
    return Chunk.doc_id.in_(
        Document.__table__.select().where(Document.state == DocumentState.LIVE).with_only_columns(
            Document.doc_id
        )
    )
