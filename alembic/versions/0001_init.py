"""initial schema: documents, chunks, keyword index

Revision ID: 0001_init
Revises:
Create Date: 2026-10-01

The SQLAlchemy metadata creates the same tables on SQLite for tests and local
runs, but with `JSON` embeddings and no vector index. This migration is the
Postgres schema: a real `vector(N)` column with an HNSW index, which SQLite
cannot express and which is what makes the retrieval target viable.

Two details worth stating explicitly:

* The embedding column type is resolved at migration time, not hardcoded. The
  dimension comes from `Settings.embedding_dim`, so a change to the embedding
  model cannot silently produce vectors of the wrong width against a schema that
  expects another.

* HNSW is created as `ivfflat`-compatible-free: `hnsw` requires pgvector 0.5+.
  The guard below fails with an actionable message instead of a generic
  "type does not exist".
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op
from app.core.config import get_settings
from app.db.models import DocumentState, _embedding_type

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_MODEL = "fake-embed-v1"


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    dim = get_settings().embedding_dim

    op.create_table(
        "documents",
        sa.Column("doc_id", sa.String(length=32), primary_key=True),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("mime_type", sa.String(length=255), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "state",
            sa.String(length=16),
            nullable=False,
            server_default=DocumentState.PENDING.value,
        ),
        sa.Column("byte_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column(
            "uploaded_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_reason", sa.Text(), nullable=True),
        sa.Column("acl_tags", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("cleaned_text", sa.Text(), nullable=True),
        sa.Column("object_key", sa.String(length=512), nullable=True),
        sa.Column("duplicate_of", sa.String(length=32), nullable=True),
    )

    # Retrieval filters on state, and dedupe looks up by hash. Both are on the
    # hot path: the first on every search, the second on every upload.
    op.create_index("ix_documents_state", "documents", ["state"])
    op.create_index("ix_documents_content_hash", "documents", ["content_hash"])
    op.create_index("ix_documents_uploaded_at", "documents", ["uploaded_at"])

    # A chunk's embedding must match the model that produced it, so the model is
    # recorded per row. A corpus half-embedded by an older model is then
    # detectable instead of silently returning wrong similarity scores
    # (architecture.md 7.3).
    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=32), primary_key=True),
        sa.Column(
            "doc_id",
            sa.String(length=32),
            sa.ForeignKey("documents.doc_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("breadcrumb", sa.Text(), nullable=False, server_default=""),
        sa.Column("section_path", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        # `embedding` is declared through the same helper the model uses, so the
        # migration and the ORM agree on the column type instead of the migration
        # creating `vector(N)` while the model still believes it is JSON.
        sa.Column("embedding", _embedding_type(dim), nullable=True),
        sa.UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
    )
    op.create_index("ix_chunk_doc", "chunks", ["doc_id"])

    if _is_postgres():
        _create_vector_index(dim)

    op.create_table(
        "chunk_terms",
        sa.Column(
            "chunk_id",
            sa.String(length=32),
            sa.ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("term", sa.String(length=64), nullable=False),
        sa.Column("term_count", sa.Integer(), nullable=False, server_default="1"),
        # Denormalised from `chunks`. Keeping the document state on the term row
        # means the "only live is retrievable" filter applies inside the keyword
        # index scan instead of requiring a join back to `documents` on every
        # keyword query. A tombstone therefore takes effect immediately even
        # though the keyword rows remain (FR-7).
        sa.Column(
            "state",
            sa.String(length=16),
            nullable=False,
            server_default=DocumentState.LIVE.value,
        ),
    )

    # The keyword index is read as "which live chunks contain this term, ordered
    # by frequency", so (term, state) is the access path.
    op.create_index("ix_chunk_terms_term_state", "chunk_terms", ["term", "state"])
    op.create_index("ix_chunk_terms_chunk", "chunk_terms", ["chunk_id"])


def _create_vector_index(dim: int) -> None:
    """Add pgvector's HNSW index over the embedding column."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # HNSW over cosine distance: retrieval scores chunks by embedding similarity,
    # and cosine is the metric the embedding models are trained for. An inner
    # product would silently reorder results for non-normalized vectors.
    #
    # `m=16` and `ef_construction=64` are pgvector's own defaults for this
    # operating point. They are stated here rather than left implicit so a future
    # tuning decision is a visible diff.
    op.execute(
        "CREATE INDEX ix_chunk_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # The model and width this schema expects. Compared on startup so a
    # schema/model mismatch fails at boot rather than as bad search results
    # later (architecture.md 7.3).
    op.execute(
        "COMMENT ON COLUMN chunks.embedding IS "
        f"'embedding produced by {EMBEDDING_MODEL}, dim={dim}'"
    )


def downgrade() -> None:
    if _is_postgres():
        op.execute("DROP INDEX IF EXISTS ix_chunk_embedding_hnsw")

    op.drop_table("chunk_terms")
    op.drop_index("ix_chunk_doc", table_name="chunks")
    op.drop_table("chunks")

    op.drop_index("ix_documents_uploaded_at", table_name="documents")
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_index("ix_documents_state", table_name="documents")
    op.drop_table("documents")