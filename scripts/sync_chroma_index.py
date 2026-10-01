"""Sync the ChromaDB vector index from authoritative chunks.

The vector store is the authoritative membership record (architecture.md 4.3).
Chroma is a *derived* index: it is a copy of chunk ids, embeddings, and enough
metadata to answer an ANN query, not the source of truth for liveness or ACL.
Nothing in the ingest path writes to Chroma, so the collection will be empty
until this script runs.

Design points:
- Text/ACL/tombstones are enforced from SQL by `ChromaVectorStore` at query
  time, but the collection should not contain tombstoned chunks at all. This
  script only upserts chunks belonging to `live` documents and actively deletes
  any ids that no longer exist in the authoritative store.
- Running this script repeatedly is idempotent and cheap enough for periodic
  rebuilds. `doc-id` restricts upserts to a single document for incremental
  catch-up.
- The `vector_store` setting need not be `chroma` to run this script; it is
  fine to use Chroma as an additional index or to rebuild it offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import session_scope
from app.retrieval.vector_store import open_chroma_collection, sync_chroma_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild/sync Chroma index from chunks")
    parser.add_argument("--doc-id", default=None, help="restrict to a single document")
    parser.add_argument("--batch-size", type=int, default=256, help="upsert batch size")
    parser.add_argument("--collection", default=None, help="override collection name")
    parser.add_argument("--chroma-path", default=None, help="override persistent path")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.collection:
        settings.chroma_collection = args.collection
    if args.chroma_path:
        settings.chroma_path = args.chroma_path

    collection = open_chroma_collection(settings)
    with session_scope() as session:
        result = sync_chroma_index(
            session,
            collection,
            dim=settings.embedding_dim,
            batch_size=args.batch_size,
            doc_id=args.doc_id,
        )

    print(f"synced {settings.chroma_path}/{settings.chroma_collection}: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
